from nebula.core.utils.locker import Locker
from collections import deque, OrderedDict
import logging
import numpy as np
from scipy.stats import linregress
from collections import defaultdict
from nebula.core.utils.helper import cosine_metric
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import AggregationEvent, UpdateNeighborEvent
from nebula.core.situationalawareness.awareness.sareputation.sareputation import ThreatCategory, PotencialThreat

class ConsistencyReputation():
    MAX_HISTORIC_SIZE = 20
    SCORE_THRESHOLD_MALICIOUS = 0.5     # Threshold to detect posible malicious nodes
    SCORE_THRESHOLD_SUSPICIOUS = 0.6
    ADVANCED_METRICS_THRESHOLD = 5
    SIMILARITY_THRESHOLD = 0.85
    # Metrics
    DEFAULT_SIMILARITY_WEIGHT = 0.6     # Default metrics
    DEFAULT_CONSISTENCY_WEIGHT = 0.15
    DEFAULT_STABILITY_WEIGHT = 0.25
    ADVANCED_SIMILARITY_WEIGHT = 0.45   # Advanced metrics
    ADVANCED_CONSISTENCY_WEIGHT = 0.2
    ADVANCED_STABILITY_WEIGHT = 0.35
    
    
    def __init__(self, config):
        self._addr = config["addr"]
        self._verbose = config["verbose"]
        self._historical_similarities: dict[str, deque[float]] = defaultdict(deque)
        self._historical_similarities_lock = Locker("historical_similarities_lock", async_lock=True)
        self._consistency_scores: dict[str, deque[float]] = defaultdict(deque)
        self._consistency_scores_lock = Locker("consistency_scores_lock", async_lock=True)
        self._suspicious_nodes: set[PotencialThreat] = set()
        self._suspicious_nodes_lock = Locker(name="suspicious_nodes_lock", async_lock=True)
        
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
        
    async def get_suspicious_nodes(self):
        async with self._suspicious_nodes_lock:
            return self._suspicious_nodes.copy()
              
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
    
    async def analize_malice(self, node, score):
        potencial_threat = None
        if score <= self.SCORE_THRESHOLD_MALICIOUS:
            potencial_threat = PotencialThreat(node, ThreatCategory.MODEL_POISSONING)
        elif score <= self.SCORE_THRESHOLD_SUSPICIOUS:
            potencial_threat = PotencialThreat(node, ThreatCategory.BAD_BEHAVIOR)
        if potencial_threat:
            async with self._suspicious_nodes_lock:
                self._suspicious_nodes.add(potencial_threat)
    
    async def evaluate(self):
        if self._verbose: logging.info("Evaluating Consistency Reputation, generating score...")
        reputation_scores = {}

        async with self._historical_similarities_lock:
            async with self._consistency_scores_lock:
                for node, history in self.hs.items():
                    if not history:
                        reputation_scores[node] = 0.0
                        continue
                    if self._verbose: logging.info(f"Node being evaluated: {node}")
                    score = self._compute_score(list(history))
                    if self._verbose: logging.info(f"Final score: {score}")
                    self.cs[node].append(score)
                    reputation_scores[node] = score

        if self._verbose:
            for node, score in reputation_scores.items():
                logging.info(f"Node {node} consistency score: {score:.4f}")
        
        return reputation_scores

    def _compute_score(self, similarities: list[float]) -> float:
        if not similarities:
            return 0.0

        # Última similitud observada
        latest_similarity = similarities[-1]

        # Base score (cuánto se acerca a la similitud esperada)
        if latest_similarity >= self.SIMILARITY_THRESHOLD:
            base_score = 1.0
        else:
            base_score = latest_similarity / self.SIMILARITY_THRESHOLD

        # Pesos por defecto
        similarity_weight = self.DEFAULT_SIMILARITY_WEIGHT
        consistency_weight = self.DEFAULT_CONSISTENCY_WEIGHT
        stability_weight = self.DEFAULT_STABILITY_WEIGHT

        temporal_consistency = 0.5
        local_stability_score = 0.5
        trend_penalty = 0.0

        final_score = base_score

        if len(similarities) >= self.ADVANCED_METRICS_THRESHOLD:
            similarity_weight = self.ADVANCED_SIMILARITY_WEIGHT
            consistency_weight = self.ADVANCED_CONSISTENCY_WEIGHT
            stability_weight = self.ADVANCED_STABILITY_WEIGHT

            # --- Consistencia temporal (varianza) ---
            var = np.var(similarities)
            temporal_consistency = 1.0 - min(var, 1.0)

            # --- Tendencia (slope) ---
            x = list(range(len(similarities)))
            slope, _, _, _, _ = linregress(x, similarities)
            if slope < 0:
                trend_penalty = min(abs(slope), 0.2)
            elif slope > 0:
                # Refuerzo positivo si la tendencia es creciente
                base_score += min(slope, 0.2)

            # --- Estabilidad local (diferencia entre actualizaciones consecutivas) ---
            diffs = [abs(similarities[i] - similarities[i - 1]) for i in range(1, len(similarities))]
            avg_fluctuation = np.mean(diffs)
            local_stability_score = 1.0 - min(avg_fluctuation, 1.0)

            # --- Cálculo final ---
            final_score = (
                similarity_weight * base_score +
                consistency_weight * temporal_consistency +
                stability_weight * local_stability_score
            )
            if self._verbose: logging.info(f"Similarity score: {similarity_weight * base_score} | Consistency score: {consistency_weight * temporal_consistency} | Stability score: {stability_weight * local_stability_score} | Tren penalty: {trend_penalty}")
            final_score = max(0.0, final_score - trend_penalty)
                    
        return round(final_score, 4)
            
