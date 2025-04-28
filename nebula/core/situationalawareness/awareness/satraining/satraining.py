import asyncio
import logging
from nebula.core.utils.locker import Locker
from nebula.core.situationalawareness.awareness.satraining.trainingpolicy.trainingpolicy import factory_training_policy
from nebula.core.situationalawareness.awareness.sareasoner import SAMComponent
from nebula.addons.functions import print_msg_box
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nebula.core.situationalawareness.awareness.sareasoner import SAReasoner, SAMComponent
    from nebula.core.eventmanager import EventManager
    
RESTRUCTURE_COOLDOWN = 5    
    
class SATraining(SAMComponent):
    def __init__(
        self,
        sam: "SAReasoner",
        addr,
        training_policy,
        weight_strategies,
        verbose
    ):
        print_msg_box(
            msg=f"Starting Training SA\nTraining policy: {training_policy}\nWeight strategies: {weight_strategies}",
            indent=2,
            title="Training SA module",
        )
        self._sam = sam
        config = {}
        config["addr"] = addr
        self._verbose = verbose
        config["verbose"] = verbose
        self._trainning_policy = factory_training_policy(training_policy, config)
        self._weight_strategies = weight_strategies

    @property
    def sam(self):
        return self._sam

    @property
    def tp(self):
        return self._trainning_policy    

    async def init(self):
        config = {}
        config["nodes"] = set(self._sam.get_nodes_known(neighbors_only=True)) 
        await self.tp.init(config)

    async def sa_component_actions(self):
        logging.info("SA Trainng evaluating current scenario")
        asyncio.create_task(self.tp.get_evaluation_results())

