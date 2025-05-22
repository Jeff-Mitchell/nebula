from enum import Enum, IntEnum

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