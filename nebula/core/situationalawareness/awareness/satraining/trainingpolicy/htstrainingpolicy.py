from nebula.core.situationalawareness.awareness.satraining.trainingpolicy.trainingpolicy import TrainingPolicy
from nebula.core.situationalawareness.awareness.satraining.trainingpolicy.trainingpolicy import factory_training_policy
from nebula.core.situationalawareness.awareness.satraining.trainingpolicy.trainingpolicy import TrainingPolicy
import logging

# "Hybrid Training Strategy"    (HTS)
class HTSTrainingPolicy(TrainingPolicy):
    TRAINING_POLICY = {
        "qds",
        "sos",
    }
    
    def __init__(self, config):
        self._addr = config["addr"]
        self._verbose = config["verbose"]
        self._training_policies : set[TrainingPolicy] = set()
        self._training_policies.add([factory_training_policy(x, config) for x in self.TRAINING_POLICY])
        
    def __str__(self):
        return "HTS"    
        
    @property
    def tps(self):
        return self._training_policies  

    async def init(self, config):
        for tp in self.tps:
            await tp.init(config)    

    async def update_neighbors(self, node, remove=False):
        return None
    
    async def get_evaluation_results(self):
        pass

    async def evaluate(self):
        nodes_to_remove = dict()
        for tp in self.tps:
            nodes_to_remove[tp] = await tp.evaluate()
        
        for tp, nodes in nodes_to_remove.items():
            logging.info(f"Training Policy: {tp}, nodes to remove: {nodes}")
            
        return None