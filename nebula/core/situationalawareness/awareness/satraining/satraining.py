import asyncio
import logging
from nebula.core.utils.locker import Locker
from nebula.core.situationalawareness.awareness.satraining.trainingpolicy.trainingpolicy import factory_training_policy
from nebula.core.situationalawareness.awareness.sareasoner import SAMComponent
from nebula.addons.functions import print_msg_box
from nebula.core.situationalawareness.awareness.sareasoner import SAReasoner, SAMComponent
from nebula.core.eventmanager import EventManager
    
RESTRUCTURE_COOLDOWN = 5    
    
class SATraining(SAMComponent):
    def __init__(
        self,
        config
    ):
        print_msg_box(
            msg=f"Starting Training SA\nTraining policy: {training_policy}",
            indent=2,
            title="Training SA module",
        )
        self._config = config
        self._sar: SAReasoner = self._config["sar"]
        tp_config = {}
        tp_config["addr"] = self._config["addr"]
        tp_config["verbose"] = self._config["verbose"]
        training_policy = self._config["training_policy"]
        self._trainning_policy = factory_training_policy(training_policy, tp_config)

    @property
    def sar(self):
        return self._sar

    @property
    def tp(self):
        return self._trainning_policy    

    async def init(self):
        config = {}
        config["nodes"] = set(self.sar.get_nodes_known(neighbors_only=True)) 
        await self.tp.init(config)

    async def sa_component_actions(self):
        logging.info("SA Trainng evaluating current scenario")
        asyncio.create_task(self.tp.get_evaluation_results())

