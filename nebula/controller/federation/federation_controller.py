from abc import ABC, abstractmethod
from fastapi import Request
from typing import Dict
from nebula.controller.federation.scenario_builder import ScenarioBuilder
import logging 

class FederationController(ABC):
    
    def __init__(self, wa_controller_url, logger):
        self._logger: logging.Logger = logger
        self._wa_url = wa_controller_url
        self._scenario_builder = ScenarioBuilder()
        
    @property
    def sb(self):
        return self._scenario_builder
    
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

