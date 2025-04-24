from nebula.core.utils.locker import Locker
from collections import deque
import logging
from collections import defaultdict
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateReceivedEvent, AggregationEvent, RoundStartEvent, UpdateNeighborEvent, NodeBlacklistedEvent
import time
from enum import Enum
import asyncio

class ThreatCategoryBehavior(Enum):
    FLOODING = "flooding"
    INACTIVITY = "inactivity"
    BAD_BEHAVIOR = "bad behavior"

class TimeStamp():
        def __init__(self, time_received = None, time_since_last_event = None):
            self.tr = time_received
            self.tsle = time_since_last_event

        def __sub__(self, other):
            if not isinstance(other, TimeStamp):
                raise TypeError("Subtraction is only supported between TimeStamp instances")   
            if self.tr is None or other.tr is None:
                raise ValueError("Cannot subtract TimeStamp instances with undefined 'tr' values")
            return self.tr - other.tr
        
        def __add__(self, other):
            if not isinstance(other, TimeStamp):
                raise TypeError("Subtraction is only supported between TimeStamp instances")   
            if self.tsle is None or other.tsle is None:
                raise ValueError("Cannot subtract TimeStamp instances with undefined 'tsle' values")
            return self.tsle + other.tsle
        
        def __str__(self):
            return f"{self.tsle}s"
        
        def is_empty(self):
            return self.tr == None
            
        def reset(self):
            self.tr = None
            self.tsle = None

class BehaviorReputation():
    MAX_HISTORIC_SIZE = 10
    SCORE_THRESHOLD = 0.5       # Threshold to detect posible malicious nodes
    MAX_MESSAGES_PER_ROUND = 15 # Maximun number os messages allowed for each node
    MAX_INACTIVITY_ALLOWED = 3  # Max number of consecutive inactive rounds
    INACTIVE_THRESHOLD = 3
    W_UPDATE_FREQ = 0.25        # Update frequency weight
    W_UPDATE_LATENCY = 0.15     # update latency weight
    W_AGG_WAITING = 0.6         # time waited since start waiting for aggregation until update is received weight
    W_INACTIVITY_PEN = 0.1      # inactivity penalty weight
     
    def __init__(self, config):
        self._addr = config["addr"]
        self._verbose = config["verbose"]
        # historic      - register of updates per round
        # inactivity    - consecutive rounds inactivity
        # t_between_upd - time between updates
        # t_last_agg    - time waited since last aggregation until receive update
        self._nodes : dict[str, tuple[deque, int, deque[TimeStamp],  deque[TimeStamp]]] = {}                                # Updates time registration
        self._nodes_lock = Locker(name="nodes_lock", async_lock=True)
        self._historical_blacklist_activity: dict[str, tuple[int,int]] = defaultdict(tuple)                                 # Blacklist activity 
        self._historical_blacklist_activity_lock = Locker(name="historical_blacklist_activity_lock", async_lock=True)
        self._historical_behavior_scores: dict[str, deque[int]] = defaultdict(deque)                                        # Scores registration
        self._messages_received_per_round: dict[str, int] = defaultdict(int)
        self._messages_received_per_round_lock = Locker(name="messages_received_per_round_lock", async_lock=True)
        self._internal_rounds_done = -1
        self._last_aggregation_time = None
        self._suspicious_nodes = set()
        self._suspicious_nodes_lock = Locker(name="suspicious_nodes_lock", async_lock=True)
        
    @property
    def hba(self):
        """historical blacklist activity"""
        return self._historical_blacklist_activity
    
    @property
    def hbs(self):
        """historical behavior scores"""
        return self._historical_behavior_scores
    
    @property
    def hba_lock(self):
        return self._historical_blacklist_activity_lock    
        
    def __str__(self):
        return "Behavior Reputation"

    async def init(self, config):
        async with self._nodes_lock:
            nodes = config["nodes"]
            self._nodes = {
                node_id: (
                    deque(maxlen=self.MAX_HISTORIC_SIZE),   # Updates per round,
                    0,                                      # Inactivity
                    deque(maxlen=self.MAX_HISTORIC_SIZE),   # Time gaps between updates 
                    deque(maxlen=self.MAX_HISTORIC_SIZE)    # Times since last aggregation
                ) for node_id in nodes
            }
            self.hbs = {
                node_id: (
                    deque(maxlen=self.MAX_HISTORIC_SIZE),   # Historical scores
                ) for node_id in nodes
            }
            
        await EventManager.get_instance().subscribe_node_event(UpdateReceivedEvent, self._process_update_received_event)
        await EventManager.get_instance().subscribe_node_event(RoundStartEvent, self._process_round_start)
        await EventManager.get_instance().subscribe_node_event(AggregationEvent, self._process_aggregation_event)
        await EventManager.get_instance().subscribe_node_event(UpdateNeighborEvent, self._update_neighbors)
        await EventManager.get_instance().subscribe_node_event(NodeBlacklistedEvent, self._process_node_blacklisted_event)
        await EventManager.get_instance().subscribe(None, self._process_messages_received)

    async def get_behavior_scores(self, historical=False):
        if historical:
            return self.hbs.copy()
        else:
            last_scores = {node: scores[-1] for node,scores in self.hbs.items()}
            return last_scores

    async def get_suspicious_nodes(self):
        async with self._suspicious_nodes_lock:
            return self._suspicious_nodes.copy()

    async def _get_nodes(self):
        async with self._nodes_lock:
            nodes = self._nodes.copy()
        return nodes
    
    def _update_behavior_scores(self, scores: dict):
        for node, score in scores.items():
            self.hbs[node].append(score)
    
    async def _process_round_start(self, rse: RoundStartEvent):
        if self._verbose: logging.info("Processing round start event")
        if not self._last_aggregation_time:
            if self._verbose: logging.info("First round start timing assigment")
            (_, start_time) = await rse.get_event_data()
            self._last_aggregation_time = start_time
        self._internal_rounds_done += 1

        async with self._messages_received_per_round_lock:
            self._messages_received_per_round.clear()

    async def _process_aggregation_event(self, are: AggregationEvent):
        self._last_aggregation_time = time.time()
        if self._verbose: logging.info("Processing aggregation event")
        (_, expected_nodes, missing_nodes) = await are.get_event_data()

        async with self._nodes_lock:
            for node in expected_nodes:
                if node in self._nodes:
                    history, missed_count, gap_btween_updts, time_since_agg = self._nodes[node]
                    self._nodes[node] = (history, 0 if node not in missing_nodes else missed_count + 1, gap_btween_updts, time_since_agg)

    async def _process_update_received_event(self, ure: UpdateReceivedEvent):
        time_received = time.time()
        if self._verbose: logging.info("Processing Update Received event")
        (_, _, source, _, _) = await ure.get_event_data()

        async with self._nodes_lock:
            if source not in self._nodes:
                return  

            history, missed_count, time_between_updts_historic, last_update_times = self._nodes[source]

            if history and history[-1][0] == self._internal_rounds_done:
                num_updates = history[-1][1] + 1
                history[-1] = (self._internal_rounds_done, num_updates)
            else:
                history.append((self._internal_rounds_done, 1))

            if not len(time_between_updts_historic):
                time_between_updts_historic.append(TimeStamp(time_received, None))
            else:
                ts = TimeStamp(time_received)
                ts.tsle = ts - time_between_updts_historic[-1]
                time_between_updts_historic.append(ts)

            lut = TimeStamp(time_received, time_received - self._last_aggregation_time)
            last_update_times.append(lut)

            self._nodes[source] = (history, missed_count, time_between_updts_historic, last_update_times)

    async def _update_neighbors(self, une: UpdateNeighborEvent):
        node, remove = await une.get_event_data()
        async with self._nodes_lock:
            if remove:
                self._nodes.pop(node, None)
            else:
                if not node in self._nodes:
                    self._nodes.update({node : (deque(maxlen=self.MAX_HISTORIC_SIZE), 0, float('inf'), float('inf'))})

    async def _process_node_blacklisted_event(self, nbe: NodeBlacklistedEvent):
        node_addr, blacklisted = await nbe.get_event_data()
        times_bl = 0
        times_rd = 0
        async with self.hba_lock:
            if self.hba[node_addr]:
                bl, rd = self.hba[node_addr]
                times_bl = bl
                times_rd = rd
            if blacklisted: # It means forced blacklisted not recently disconnected
                times_bl += 1
            else:
                times_rd += 1
            self.hba[node_addr] = (times_bl, times_rd)

    async def _process_messages_received(self, source, message):
        async with self._messages_received_per_round_lock:
            n_messages = self._messages_received_per_round[source]
            n_messages += 1
            if n_messages >= self.MAX_MESSAGES_PER_ROUND:
                async with self._suspicious_nodes_lock:
                    self._suspicious_nodes.union({(source, ThreatCategoryBehavior.FLOODING)})
            self._messages_received_per_round[source] = n_messages
            
    async def _evaluate(self):
        if self._verbose: logging.info("Evaluating Behavior Reputation, generating score...")

        nodes = await self._get_nodes()
        for node in nodes.keys():
            #logging.info(f"Node: {node}, {nodes[node][0]}")
            updates_received = {x[1] for x in nodes[node][0] if x[0] == self._internal_rounds_done}
            if self._verbose: logging.info(f"Node: {node} | Updates received this round: {updates_received}")
            if self._verbose: logging.info(f"Time waited since last aggregation event {nodes[node][3][-1].tsle:.3f}")
            
        # Extract max and min values for normalization
        max_updates = max(
        (
            max((x[1] for x in nodes[n][0] if x[0] == self._internal_rounds_done), default=0)
            for n in nodes
        ),
        default=1
        )

        # Min latency observed
        min_latency = min(
            (
                sum(t.tsle for t in nodes[n][2] if t.tsle is not None and t.tsle != float('inf')) / len(nodes[n][2])
                if any(t.tsle is not None and t.tsle != float('inf') for t in nodes[n][2])
                else float('inf')
                for n in nodes
            ),
            default=1
        )

        # Min wait time observed
        min_wait_time = min(
            (
                sum(t.tsle for t in nodes[n][3]) / len(nodes[n][3]) if nodes[n][3] else float('inf')
                for n in nodes
            ),
            default=1
        )

        if self._verbose: logging.info(f"max updates: {max_updates} | mean min latency: {min_latency:.3f} | mean min wait time: {min_wait_time:.3f}")
        scores = {}

        for node, (history, missed_count, time_between_updts_historic, last_wait_times) in nodes.items():

            # Check inactivity beyond max tolerance
            if missed_count >= self.MAX_INACTIVITY_ALLOWED:
                async with self._suspicious_nodes_lock:
                    self._suspicious_nodes.union(((node, ThreatCategoryBehavior.INACTIVITY)))
                    scores[node] = 0.0
                    continue

            # 1. Normalized frequency
            updates_received = max((x[1] for x in history if x[0] == self._internal_rounds_done), default=0)
            F_updt_freq = updates_received / max_updates if max_updates > 0 else 0

            # 2. Normalized mean latency between updates
            valid_latencies = [t.tsle for t in time_between_updts_historic if t.tsle is not None and t.tsle != float('inf')]
            avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else float('inf')
            F_updt_latency = min_latency / avg_latency if avg_latency > 0 and avg_latency != float('inf') else 0

            # 3. Normalized time since last aggregation
            avg_wait_time = sum(t.tsle for t in last_wait_times) / len(last_wait_times) if last_wait_times else float('inf')
            F_agg_waiting = min_wait_time / avg_wait_time if avg_wait_time > 0 else 0

            # 4. Inactivity penalty
            P_n = missed_count*self.W_INACTIVITY_PEN

            # 5. Final score
            score = (
                (self.W_UPDATE_FREQ * F_updt_freq) +
                (self.W_UPDATE_LATENCY * F_updt_latency) +
                (self.W_AGG_WAITING * F_agg_waiting) -
                P_n
            )
            scores[node] = score
        
        self._update_behavior_scores(scores)
        
        # Final Step
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        nodes_below_th = [x for x in sorted_nodes if x[1] < self.SCORE_THRESHOLD]

        if self._verbose:
            for node, score in sorted_nodes:
                 if self._verbose: logging.info(f"Node: {node} | Score: {score:.3f}")
                
        if self._verbose: logging.info(f"Nodes below threshold: {nodes_below_th}")
        
        # Update suspicious nodes
        async with self._suspicious_nodes_lock:
            self._suspicious_nodes.union({(n, ThreatCategoryBehavior.BAD_BEHAVIOR) for n in nodes_below_th}) 
                                       
    
    
