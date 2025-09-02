from abc import ABC, abstractmethod
from fastapi import Request
from typing import Dict
from nebula.controller.federation.scenario_builder import ScenarioBuilder
import logging 

class NebulaFederation(ABC):
    pass

class FederationController(ABC):
    
    def __init__(self, hub_url, logger):
        self._logger: logging.Logger = logger
        self._hub_url = hub_url

    @property
    def logger(self):
        return self._logger 

    @abstractmethod
    async def run_scenario(self,  id: str, scenario_data: Dict, user: str):
        pass

    @abstractmethod
    async def stop_scenario(self, id: str):
        pass

    @abstractmethod
    async def update_nodes(self, scenario_name: str, request: Request):
        pass
    
    abstractmethod
    async def node_done(self, scenario_name: str, request: Request):
        pass
