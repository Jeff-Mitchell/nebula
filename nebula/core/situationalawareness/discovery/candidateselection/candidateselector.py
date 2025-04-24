from abc import ABC, abstractmethod
from typing import Type

class CandidateSelector(ABC):
    
    @abstractmethod 
    def set_config(self, config):
        pass
    
    @abstractmethod 
    def add_candidate(self, candidate):
        pass
    
    @abstractmethod 
    def select_candidates(self):
        pass
    
    @abstractmethod 
    def remove_candidates(self):
        pass
    
    @abstractmethod 
    def any_candidate(self):
        pass
    
def factory_CandidateSelector(topology) -> CandidateSelector:
    from nebula.core.situationalawareness.discovery.candidateselection.stdcandidateselector import STDandidateSelector
    from nebula.core.situationalawareness.discovery.candidateselection.fccandidateselector import FCCandidateSelector
    from nebula.core.situationalawareness.discovery.candidateselection.hetcandidateselector import HETCandidateSelector
    from nebula.core.situationalawareness.discovery.candidateselection.ringcandidateselector import RINGCandidateSelector
    
    options = {
        "ring": RINGCandidateSelector,
        "fully": FCCandidateSelector,
        "random": STDandidateSelector,
        "het": HETCandidateSelector,  
    } 
    
    cs = options.get(topology, FCCandidateSelector)
    return cs() 