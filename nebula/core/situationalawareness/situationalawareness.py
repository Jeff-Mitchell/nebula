from nebula.core.situationalawareness.discovery.federationconnector import FederationConnector
from nebula.core.situationalawareness.awareness.samodule import SAModule
from abc import ABC, abstractmethod

class ISADiscovery(ABC):
    @abstractmethod
    async def start_late_connection_process(self, connected=False, msg_type="discover_join", addrs_known=None):
        raise NotImplementedError

class ISAReasoner(ABC):
    @abstractmethod
    def accept_connection(self, source, joining=False):
        raise NotImplementedError
    
    @abstractmethod
    def get_nodes_known(self, neighbors_too=False, neighbors_only=False):
        raise NotImplementedError
    
    @abstractmethod
    def get_actions(self):
        raise NotImplementedError

class SituationalAwareness():
    def __init__(self):
        #self._situational_awareness_module = SAModule(self, self.config, self.engine.addr, topology, True)
        pass

    async def init(self):
        pass
