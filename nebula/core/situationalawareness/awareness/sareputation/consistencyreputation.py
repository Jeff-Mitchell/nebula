from nebula.core.utils.locker import Locker
from collections import deque, OrderedDict
import logging
import numpy as np
from scipy.stats import linregress
from collections import defaultdict
from nebula.core.utils.helper import cosine_metric
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import AggregationEvent, UpdateNeighborEvent

class ConsistencyReputation():
    MAX_HISTORIC_SIZE = 20
    CONSISTENCY_THRESHOLD = 5
    SIMILARITY_THRESHOLD = 0.85
    DEFAULT_SIMILARITY_WEIGHT = 0.7
    DEFAULT_CONSISTENCY_WEIGHT = 0.3
    ADVANCED_SIMILARITY_WEIGHT = 0.6
    ADVANCED_CONSISTENCY_WEIGHT = 0.4
    
    def __init__(self, config):
        self._addr = config["addr"]
        self._verbose = config["verbose"]
        self._historical_similarities: dict[str, deque[float]] = defaultdict(deque)
        self._historical_similarities_lock = Locker("historical_similarities_lock", async_lock=True)
        self._consistency_scores: dict[str, deque[float]] = defaultdict(deque)
        self._consistency_scores_lock = Locker("consistency_scores_lock", async_lock=True)
        
    @property
    def hs(self):
        return self._historical_similarities
    
    @property
    def cs(self):
        return self._consistency_scores
        
    async def init(self, neighbor_list):
        async with self._historical_similarities_lock:
            async with self._consistency_scores_lock:
                for node in neighbor_list:
                    self.hs[node] = deque(maxlen=self.MAX_HISTORIC_SIZE)
                    self.cs[node] = deque(maxlen=self.MAX_HISTORIC_SIZE)
                
        await EventManager.get_instance().subscribe_node_event(AggregationEvent, self._process_aggregation_event)
        await EventManager.get_instance().subscribe_node_event(UpdateNeighborEvent, self._process_update_neighbor_event)
              
    async def _process_update_neighbor_event(self, une: UpdateNeighborEvent):
        node, remove = await une.get_event_data()
        async with self._historical_similarities_lock:
            async with self._consistency_scores_lock:
                if remove:
                    self.hs.pop(node, None)
                    self.cs.pop(node, None)
                else:
                    if not node in self.hs:
                        self.hs.update({node : deque(maxlen=self.MAX_HISTORIC_SIZE)})
                        self.cs.update({node : deque(maxlen=self.MAX_HISTORIC_SIZE)})
    
    async def _process_aggregation_event(self, age: AggregationEvent):
        (updates, expected_nodes, _) = await age.get_event_data()
        self_model, _ = updates[self._addr]
        async with self._historical_similarities_lock:
            for node in expected_nodes:
                similarity = self._calculate_model_similarity(self_model, updates[node][0])
                self.hs[node].append(similarity)

    async def get_scores(self, historical=False):
        if historical:
            return self.cs.copy()
        else:
            last_scores = {node: scores[-1] for node,scores in self.cs.items()}
            return last_scores
        
    def _calculate_model_similarity(self, model1: OrderedDict, model2: OrderedDict):
        return cosine_metric(model1=model1, model2=model2, similarity=True)
    
    async def evaluate(self):
        if self._verbose: logging.info("Evaluating Consistency Reputation, generating score...")
        reputation_scores = {}

        async with self._historical_similarities_lock:
            async with self._consistency_scores_lock:
                for node, history in self.hs.items():
                    if not history:
                        reputation_scores[node] = 0.0
                        continue
                    
                    score = self._compute_score(list(history))
                    self.cs[node].append(score)
                    reputation_scores[node] = score

        if self._verbose:
            for node, score in reputation_scores.items():
                logging.info(f"Node {node} consistency score: {score:.4f}")
        
        return reputation_scores

    def _compute_score(self, similarities: list[float]) -> float:
        if not similarities:
            return 0.0
        
        similarity_weight = self.DEFAULT_SIMILARITY_WEIGHT
        consistency_weight = self.DEFAULT_CONSISTENCY_WEIGHT

        latest_similarity = similarities[-1]

        if latest_similarity >= self.SIMILARITY_THRESHOLD:
            base_score = 1.0
        else:
            # Linearly scaled score between 0 and 1 based on how close it is to the threshold
            base_score = latest_similarity / self.SIMILARITY_THRESHOLD

        temporal_consistency = 0.5  # Default medium trust for very short histories

        # Temporal consistency: compute inverse variance (lower variance → higher trust)
        if len(similarities) >= self.CONSISTENCY_THRESHOLD:
            similarity_weight = self.ADVANCED_SIMILARITY_WEIGHT
            consistency_weight = self.ADVANCED_CONSISTENCY_WEIGHT
        
            var = np.var(similarities)
            temporal_consistency = 1.0 - min(var, 1.0)
            
             # 2. Trend analysis: penalize downward trends
            x = list(range(len(similarities)))
            slope, _, _, _, _ = linregress(x, similarities)

            # Normalize slope to a range [-1, 1] depending on steepness
            # and penalize negative slope
            if slope < 0:
                # For example: max penalty of 0.2 if strong negative slope
                trend_penalty = min(abs(slope), 0.2)
            

        # Weighted average: 70% recent similarity, 30% temporal consistency
        final_score = similarity_weight * base_score + consistency_weight * temporal_consistency
        return round(final_score, 4)