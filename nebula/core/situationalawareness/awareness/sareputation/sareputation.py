from nebula.core.situationalawareness.awareness.sareasoner import SAMComponent
from enum import Enum, IntEnum
from nebula.config.config import Config
from nebula.addons.functions import print_msg_box
from nebula.core.network.communications import CommunicationsManager
from nebula.core.situationalawareness.awareness.sautils.samoduleagent import SAModuleAgent
from nebula.core.situationalawareness.awareness.suggestionbuffer import SuggestionBuffer
from collections import deque, defaultdict
from nebula.core.utils.locker import Locker
import logging
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


"""                                                     ##############################
                                                        #           THREATS          #
                                                        ##############################
"""

class ThreatCategory(Enum):
    FLOODING = "flooding"
    INACTIVITY = "inactivity"
    BAD_BEHAVIOR = "bad behavior"
    MODEL_POISSONING = "model poissoning"
    
class PotencialThreat():
    def __init__(self, source, threat: ThreatCategory):
        self.source = source
        self.threat = threat
        
    def __eq__(self, other):
        if not isinstance(other, PotencialThreat):
            return False
        return (self.source == other.source and
                self.threat == other.threat)

    def __hash__(self):
        return hash((self.source, self.threat))
    
    def __str__(self):
        return f"Node: {self.source} got a potential threat: {self.threat.value}"
    
"""                                                     ##############################
                                                        #         REPUTATION         #
                                                        ##############################
"""

class ReputationCategory(IntEnum):
    """
    Enumeration representing the reputation categories of a node or entity.
    Provides conversion from string labels and numerical scores.
    """
    
    MALICIOUS = 0
    SUSPICIOUS = 1
    RESPECTED = 2
    TRUSTED = 3
    HIGH_TRUSTED = 4

    _LABELS = {
        HIGH_TRUSTED: "high_trusted",
        TRUSTED: "trusted",
        RESPECTED: "respected",
        SUSPICIOUS: "suspicious",
        MALICIOUS: "malicious",
    }

    _LABEL_TO_CATEGORY = {label: cat for cat, label in _LABELS.items()}
    
    THRESHOLDS = {
            HIGH_TRUSTED: 0.9,
            TRUSTED: 0.8,
            RESPECTED: 0.6,
            SUSPICIOUS: 0.5,
            MALICIOUS: 0.0,
        }

    @property
    def label(self) -> str:
        """Returns the string label associated with this category."""
        return self._LABELS[self]
    
    @classmethod
    def reputation_category(cls, rep_label: str) -> 'ReputationCategory':
        """Converts a string label to its corresponding ReputationCategory."""
        return _LABEL_TO_CATEGORY.get(rep_label)
    
    @classmethod
    def from_score(cls, score: float) -> 'ReputationCategory':
        """
        Returns the appropriate ReputationCategory for a given numeric reputation score.
        Categories are selected by descending threshold values.
        """
        for category, threshold in sorted(_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return category
            
# Auxiliary dictionaries declared outside the Enum
_LABELS = {
    ReputationCategory.HIGH_TRUSTED: "high_trusted",
    ReputationCategory.TRUSTED: "trusted",
    ReputationCategory.RESPECTED: "respected",
    ReputationCategory.SUSPICIOUS: "suspicious",
    ReputationCategory.MALICIOUS: "malicious",
}

_LABEL_TO_CATEGORY = {label: cat for cat, label in _LABELS.items()}

_THRESHOLDS = {
    ReputationCategory.HIGH_TRUSTED: 0.9,
    ReputationCategory.TRUSTED: 0.8,
    ReputationCategory.RESPECTED: 0.6,
    ReputationCategory.SUSPICIOUS: 0.5,
    ReputationCategory.MALICIOUS: 0.0,
}
        
class ReputationScore():
    def __init__(self, target, reputation_score):
        self._target = target
        self._reputation = reputation_score
        self._reputation_category = ReputationCategory.from_score(reputation_score)
             
    @property
    def target(self):
        return self._target

    @property
    def category(self):
        return self._reputation_category

    @property
    def reputation(self):
        return self._reputation
    
    def update_reputation(self, reputation_score):
        self._reputation = reputation_score
        self._reputation_category = ReputationCategory.from_score(reputation_score)
       
"""                                                     ##############################
                                                        #       SA REPUTATION        #
                                                        ##############################
"""       
        
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
                self.reps.pop(node, None)
            else:
                if not node in self.reps.keys():
                    self.reps.update({node : deque(maxlen=self.MAX_HISTORIC_SIZE)})