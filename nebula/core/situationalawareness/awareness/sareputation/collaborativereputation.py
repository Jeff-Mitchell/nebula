from nebula.core.utils.locker import Locker
from collections import deque, defaultdict, Counter
import logging
import asyncio
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateNeighborEvent
from nebula.core.situationalawareness.awareness.sareputation.sareputation import ThreatCategory, ReputationCategory, reputation_category
from nebula.core.network.communications import CommunicationsManager
import numpy as np

class Trial():
    TRIAL_DURATION = 20
    def __init__(self, defendant, n_jury):
        self._defendant = defendant
        self._jury: int = n_jury
        self._verdicts: list[ReputationCategory] = list()
        self._verdicts_lock = Locker("verdicts_lock", async_lock=True)
        self._final_judgment = asyncio.get_event_loop().create_future()
        self._trial_duration = None
        
    async def init_trial(self, timeout=None):
        self._trial_duration = timeout if timeout else self.TRIAL_DURATION
        
    async def force_stop_trial(self):
        async with self._verdicts_lock:
            self._final_judgment.set_result(None)
        
    async def _close_trial(self):
        asyncio.sleep(self._trial_duration)
        async with self._verdicts_lock:
            await self._make_sentence()
    
    async def receive_verdict(self, verdict: ReputationCategory):
        async with self._verdicts_lock:
            self._verdicts.append(verdict)
            n_verdicts = len(self._verdicts)
        if n_verdicts == self._jury:
            await self._make_sentence()
            
    async def _make_sentence(self):
        vote_count = Counter(self._verdicts)
        if not vote_count:
            self._final_judgment.set_result(None)
            return

        max_votes = max(vote_count.values())
        top_categories = [cat for cat, count in vote_count.items() if count == max_votes]       # Draw situation
        final_category = min(top_categories)                                                    # Get lower category 
        self._final_judgment.set_result(final_category) 
            
    async def get_judgment(self) -> asyncio.Future:
        return self._final_judgment
            
class TrialPolicy():
    MAX_TRIALS_ACCEPTED = 3
    TRIAL_TIMEOUT = 20
    def __init__(self):
        self._max_trials = self.MAX_TRIALS_ACCEPTED
        self._trial_timeout = self.TRIAL_TIMEOUT
        self._trials_recently_accepted = 0
        self._trials_lock = Locker("trials_lock", async_lock=True)
    
    async def _clean_trial_count(self):
        asyncio.sleep(self.TRIAL_TIMEOUT)
        async with self._trials_lock:
            self._trials_recently_accepted -= 1
        
    async def accept_trial_request(self, source):
        async with self._trials_lock:
            if self._trials_recently_accepted < self.MAX_TRIALS_ACCEPTED:
                self._trials_recently_accepted += 1
                asyncio.create_task(self._clean_trial_count())
                return True
            else:
                return False
                
class CollaborativeReputation():
    MAX_HISTORIC_SIZE = 10
    
    def __init__(self):
        self._trials_recently_received: int = 0
        self._nodes_on_trial = set()
        self._social_expectations: dict[str, deque[ReputationCategory]] = defaultdict(deque)
        self._social_expectations_lock = Locker("social_expectations_lock", async_lock=True)
        self._social_reputations: dict[str, ReputationCategory] = defaultdict(ReputationCategory)
        self._social_reputations_lock = Locker("social_reputations", async_lock=True)
        self._trial_policy = TrialPolicy()
        self._open_trials: dict[str, Trial] = defaultdict(Trial)
        self._open_trials_lock = Locker("open_trials_lock", async_lock=True)
        self._verbose = True
        
    @property
    def se(self):
        """Social Expectations"""
        return self._social_expectations
    
    @property
    def sr(self):
        """Social Reputations"""
        return self._social_reputations
    
    @property
    def tp(self):
        """Trial Policy"""
        return self._trial_policy
    
    @property
    def ot(self):
        """Open Trials"""
        return self._open_trials
        
    async def init(self, neighbor_list):
        for node in neighbor_list:
            self.se[node] = deque(maxlen=self.MAX_HISTORIC_SIZE)
        await EventManager.get_instance().subscribe(("reputation", "share_reputation"), self._process_share_reputation_message)
        await EventManager.get_instance().subscribe(("reputation", "start_trial"), self._process_start_trial_message)
        await EventManager.get_instance().subscribe(("reputation", "submit_verdict"), self._process_submit_verdict_message)
        await EventManager.get_instance().subscribe_node_event(UpdateNeighborEvent, self._process_update_neighbor_event)
    
    """                                                 ##################################
                                                        # REPUTATION MESSAGES PROCESSING #
                                                        ##################################
    """

    async def _process_update_neighbor_event(self, une: UpdateNeighborEvent):
        node, remove = await une.get_event_data()
        async with self._social_expectations_lock:
                if remove:
                    self.se.pop(node, None)
                    async with self._open_trials_lock:
                        if node in self.ot:
                            await self.ot[node].force_stop_trial()
                else:
                    if not node in self.se.keys():
                        self.se.update({node : deque(maxlen=self.MAX_HISTORIC_SIZE)})
        
    async def _process_share_reputation_message(self, source, message):
        rep_value = message.reputation 
        rep_category = reputation_category(rep_value)
        async with self._social_expectations_lock:
            self.se[source].append(rep_category)    
    
    async def _process_start_trial_message(self, source, message):
        if await self.tp.accept_trial_request(source):
            defendant = message.defendant
            if self._verbose: logging.info(f"Processing trial request from: {source}, defendant: {defendant}")
            async with self._social_reputations_lock:
                defendant_reputation = self.sr.get(defendant, None)
            verdict_message = CommunicationsManager.get_instance().create_message("reputation", "submit_verdict", defendant=defendant, verdict=defendant_reputation.value)
            asyncio.create_task(CommunicationsManager.get_instance().send_message(source, verdict_message))
        else:
            if self._verbose: logging.info(f"Ignoring trial request from: {source}...")
            
    async def _process_submit_verdict_message(self, source, message):
        verdict = message.verdict
        verdict = reputation_category(verdict)
        async with self._open_trials_lock:
            if source in self.ot:
                await self.ot.get(source).receive_verdict(verdict)
            else:
                if self._verbose: logging.info(f"There is no open trial on source: {source}")
                
    async def _start_trial_to_node(self, node):
        start_trial_message = CommunicationsManager.get_instance().create_message("reputation", "start_trial", defendant=node)
        await CommunicationsManager.get_instance().send_message_to_neighbors(start_trial_message)
            
    async def update_social_reputations(self, reputations: dict[str, ReputationCategory]):
        async with self._social_reputations_lock:
            self._social_reputations = reputations
    
    """                                                 ##################################
                                                        #         FUNCTIONALITIES        #
                                                        ##################################
    """    
            
    async def close_trial(self, node):
        async with self._open_trials_lock:
            self.ot.pop(node)
            
    async def start_trial(self, node):
        opened = False
        async with self._open_trials_lock:
            if not node in self.ot:
                new_trial = Trial(node)
                self.ot[node] = new_trial
                await new_trial.init_trial()
                opened = True
        if opened: await self._start_trial_to_node(node)
            
    async def get_social_state(self):
        """Function to calculate social suitability state and suitability score"""
        suitable = False
        suitable_score = 0
        async with self._social_expectations_lock:           
            last_reputations_received = [rep[-1] for rep in self.se.values() if rep and len(rep) > 0]
            
        if last_reputations_received:
            suitable = all(lrr > ReputationCategory.TRUSTED.value for lrr in last_reputations_received)
            if suitable:
                suitable_score = np.mean(last_reputations_received)
                return (suitable, suitable_score)
        else:
            return (False, 0)