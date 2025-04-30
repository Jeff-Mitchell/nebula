from nebula.core.situationalawareness.discovery.federationconnector import FederationConnector
from nebula.core.situationalawareness.awareness.sareasoner import SAReasoner
from abc import ABC, abstractmethod
from nebula.addons.functions import print_msg_box

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
    def __init__(self, config):
        print_msg_box(
            msg=f"Starting Situational Awareness module...",
            indent=2,
            title="Situational Awareness module",
        )
        self._config = config
        topology = self._config.participant["mobility_args"]["topology_type"]
        topology = topology.lower()
        model_handler = "std" 
        self._federation_connector = FederationConnector(
            self._config.participant["mobility_args"]["additional_node"]["status"],
            topology,
            model_handler,
            engine=self,
            verbose=True
        )
        self._sareasoner = SAReasoner(
            self._config, 
            self._config.participant["network_args"]["addr"], 
            topology, 
            verbose=True
        )
    
    @property
    def fedcon(self):
        """Federation Connector"""
        return self._federation_connector
    
    @property
    def sar(self):
        """SA Reasoner"""
        return self._sareasoner

    async def init(self):
        await self.fedcon.init(self.sar)
        await self.sar.init(self.fedcon)
    
    async def start_late_connection_process(self):
        await self.fedcon.start_late_connection_process()
        
    async def get_trainning_info(self):
        return await self.fedcon.get_trainning_info()
