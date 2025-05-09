from abc import ABC, abstractmethod
from typing import Type

class NeighborPolicy(ABC):
    
    @abstractmethod 
    def set_config(self, config):
        pass
    
    @abstractmethod 
    def need_more_neighbors(self):
        pass

    @abstractmethod 
    def any_leftovers_neighbors(self):
        pass

    @abstractmethod 
    def get_neighbors_to_remove(self):
        pass

    @abstractmethod
    def accept_connection(self, source, joining=False):
        pass
    
    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def meet_node(self, node):
        pass
    
    abstractmethod
    def forget_nodes(self, nodes, forget_all=False):
        pass
    
    @abstractmethod
    def get_nodes_known(self, neighbors_too=False, neighbors_only=False):
        pass
    
    @abstractmethod
    def update_neighbors(self, node, remove=False):
        pass

    @abstractmethod
    def stricted_topology_status(stricted_topology: bool):
        pass

def factory_NeighborPolicy(topology) -> NeighborPolicy:
    from nebula.core.situationalawareness.awareness.sanetwork.neighborpolicies.idleneighborpolicy import IDLENeighborPolicy
    from nebula.core.situationalawareness.awareness.sanetwork.neighborpolicies.fcneighborpolicy import FCNeighborPolicy
    from nebula.core.situationalawareness.awareness.sanetwork.neighborpolicies.ringneighborpolicy import RINGNeighborPolicy
    from nebula.core.situationalawareness.awareness.sanetwork.neighborpolicies.starneighborpolicy import STARNeighborPolicy
    
    options = {
        "random": IDLENeighborPolicy, # default value
        "fully": FCNeighborPolicy,
        "ring": RINGNeighborPolicy,
        "star": IDLENeighborPolicy,
    } 
    
    cs = options.get(topology, IDLENeighborPolicy)
    return cs() 