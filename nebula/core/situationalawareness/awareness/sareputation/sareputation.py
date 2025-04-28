from nebula.core.situationalawareness.awareness.sareasoner import SAMComponent
from enum import Enum

class ReputationCategory(Enum): # Reputational thresholds
    HIGH_TRUSTED = 0.9
    TRUSTED = 0.8
    RESPECTED = 0.6
    SUSPICIOUS = 0.5
    MALICIOUS = 0.0

class ReputationScore():
    
    def __init__(self, target, reputation_score):
        self._target = target
        self._reputation = reputation_score
        self._reputation_category = self._assign_reputation_category(reputation_score)
        
    def _assign_reputation_category(self, score):
        for category in sorted(ReputationCategory, key=lambda c: c.value, reverse=True):
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