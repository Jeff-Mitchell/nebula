from nebula.core.situationalawareness.awareness.sareasoner import SAMComponent
from nebula.config.config import Config
from nebula.addons.functions import print_msg_box
from nebula.core.network.communications import CommunicationsManager
from nebula.core.situationalawareness.awareness.sautils.samoduleagent import SAModuleAgent
from nebula.core.situationalawareness.awareness.suggestionbuffer import SuggestionBuffer
from collections import deque, defaultdict
from nebula.core.utils.locker import Locker
import logging
from nebula.core.situationalawareness.awareness.sareputation.utils import ReputationScore
from nebula.core.situationalawareness.awareness.sareputation.consistencyreputation import ConsistencyReputation
from nebula.core.situationalawareness.awareness.sareputation.collaborativereputation import CollaborativeReputation
from nebula.core.situationalawareness.awareness.sautils.sacommand import (
    SACommand,
    SACommandAction,
    SACommandPRIO,
    SACommandState,
    factory_sa_command,
)
from nebula.core.nebulaevents import (
    UpdateNeighborEvent,
    AggregationEvent,
)
             
# TODO create ReputationComponents allowing differents setups depending on
# the user prefs
class SAReputation(SAMComponent):
    MAX_HISTORIC_SIZE = 20
    
    def __init__(self, config: Config):
        print_msg_box(
            msg=f"Starting Reputation SA",
            indent=2,
            title="Reputation SA module",
        )
        self._config = config
        self._reputations: dict[str, deque[ReputationScore]] = defaultdict(deque)    # Only maintain last ReputationScore calculated
        self._reputations_lock = Locker("reputations_lock", async_lock=True)
        self._cm = CommunicationsManager.get_instance()
        self._rep_collaborative = CollaborativeReputation()
        self._rep_consistency = ConsistencyReputation(config={"addr": config.participant["network_args"]["addr"]})
        
    @property
    def cm(self):
        return self._cm
    
    @property
    def reps(self):
        return self._reputations
    
    @property
    def colrep(self):
        return self._rep_collaborative
    
    @property
    def conrep(self):
        return self._rep_consistency

    async def init(self):
        neighbors = await self.cm.get_addrs_current_connections(only_direct=True, myself=False)
        for node in neighbors:
            self.reps[node] = deque(maxlen=self.MAX_HISTORIC_SIZE)
        await self.colrep.init(neighbor_list=neighbors)
        await self.conrep.init(neighbor_list=neighbors)
    
    async def sa_component_actions(self):
        async with self._reputations_lock:
            last_reputations = {node: reps[-1].category.label for node, reps in self.reps.items() if len(reps) > 0}
            await self.colrep.update_social_reputations(last_reputations)
            await self.colrep.share_reputations()
    
    # --- Events processing ---
    async def _process_update_neighbor_event(self, une: UpdateNeighborEvent):
        #TODO if we forget the disc ones, attacks connect/disc could f*me up
        node, remove = await une.get_event_data()
        async with self._reputations_lock:
            if remove:
                #TODO verify open trials
                self.reps.pop(node, None)
            else:
                if not node in self.reps.keys():
                    self.reps.update({node : deque(maxlen=self.MAX_HISTORIC_SIZE)})