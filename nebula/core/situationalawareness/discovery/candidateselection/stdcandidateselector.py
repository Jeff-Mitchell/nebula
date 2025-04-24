from nebula.core.situationalawareness.discovery.candidateselection.candidateselector import CandidateSelector
from nebula.core.utils.locker import Locker

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
        #TODO revisar
        mean_neighbors = sum(n for n, _ in self.candidates) / len(self.candidates) if self.candidates else 0
        cdts = self.candidates[:mean_neighbors]
        self.candidates_lock.release()
        return cdts
    
    def remove_candidates(self):
        self.candidates_lock.acquire()
        self.candidates = []
        self.candidates_lock.release()

    def any_candidate(self):
        self.candidates_lock.acquire()
        any = True if len(self.candidates) > 0 else False
        self.candidates_lock.release()
        return any