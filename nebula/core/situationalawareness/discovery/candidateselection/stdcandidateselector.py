from nebula.core.situationalawareness.discovery.candidateselection.candidateselector import CandidateSelector
from nebula.core.utils.locker import Locker
import logging

class STDandidateSelector(CandidateSelector):
    
    def __init__(self):
        self.candidates = []
        self.candidates_lock = Locker(name="candidates_lock")
        
    def set_config(self, config):
        pass    
    
    def add_candidate(self, candidate):
        self.candidates_lock.acquire()
        self.candidates.append(candidate)
        self.candidates_lock.release()
      
    def select_candidates(self):
        """
            Select mean number of neighbors
        """
        self.candidates_lock.acquire()
        mean_neighbors = round(sum(n for _, n, _ in self.candidates) / len(self.candidates) if self.candidates else 0)
        logging.info(f"mean number of neighbors: {mean_neighbors}")
        cdts = self.candidates[:mean_neighbors]
        not_selected = set(self.candidates) - set(cdts)
        self.candidates_lock.release()
        return (cdts, not_selected)
    
    def remove_candidates(self):
        self.candidates_lock.acquire()
        self.candidates = []
        self.candidates_lock.release()

    def any_candidate(self):
        self.candidates_lock.acquire()
        any = True if len(self.candidates) > 0 else False
        self.candidates_lock.release()
        return any