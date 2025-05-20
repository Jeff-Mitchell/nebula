from nebula.core.situationalawareness.awareness.sareasoner import SAMComponent
from enum import Enum

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
    
"""                                                     ##############################
                                                        #         REPUTATION         #
                                                        ##############################
"""

class ReputationCategory(Enum):
    HIGH_TRUSTED = "high_trusted"
    TRUSTED = "trusted"
    RESPECTED = "respected"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"

class ReputationThreshold(Enum): # Reputational thresholds
    HIGH_TRUSTED = 0.9
    TRUSTED = 0.8
    RESPECTED = 0.6
    SUSPICIOUS = 0.5
    MALICIOUS = 0.0
    
@staticmethod
def reputation_category(rep_cat: str):
    rep_category = next((cat for cat in ReputationCategory if cat.value == rep_cat), None)
    return rep_category

class ReputationScore():
    def __init__(self, target, reputation_score):
        self._target = target
        self._reputation = reputation_score
        self._reputation_category = self._assign_reputation_category(reputation_score)
        
    def _assign_reputation_category(self, score):
        for category in sorted(ReputationThreshold, key=lambda c: c.value, reverse=True):
            if score >= category.value:
                return category
            
    def get_category(self):
        return self._reputation_category
    
    def get_reputational_score(self):
        return self._reputation
    
    def update_reputation(self, reputation_score):
        self._reputation = reputation_score
        self._reputation_category = self._assign_reputation_category(reputation_score)
        
class SAReputation(SAMComponent):

    async def init(self):
        pass
    async def sa_component_actions(self):
        pass
    