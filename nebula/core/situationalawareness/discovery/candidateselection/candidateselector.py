from abc import ABC, abstractmethod
from typing import Type

class CandidateSelector(ABC):
    
    @abstractmethod 
    def set_config(self, config):
        """
        Configure internal parameters for the candidate selection strategy.
        
        Parameters:
            config: A configuration object or dictionary with necessary parameters.
        """
        pass
    
    @abstractmethod 
    def add_candidate(self, candidate):
        """
        Add a new candidate to the internal pool of potential selections.
        
        Parameters:
            candidate: The candidate node or object to be considered for selection.
        """
        pass
    
    @abstractmethod 
    def select_candidates(self):
        """
        Apply the selection logic to choose the best candidates from the internal pool.
        
        Returns:
            list: A list of selected candidates based on the implemented strategy.
        """
        pass
    
    @abstractmethod 
    def remove_candidates(self):
        """
        Remove one or more candidates from the pool based on internal rules or external decisions.
        """
        pass
    
    @abstractmethod 
    def any_candidate(self):
        """
        Check whether there are any candidates currently available in the internal pool.
        
        Returns:
            bool: True if at least one candidate is available, False otherwise.
        """
        pass
    
def factory_CandidateSelector(selector) -> CandidateSelector:
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
    
    cs = options.get(selector, FCCandidateSelector)
    return cs() 