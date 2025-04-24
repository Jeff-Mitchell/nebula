import logging
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

import numpy as np

from nebula.core.utils.helper import cosine_metric
from nebula.core.utils.locker import Locker

if TYPE_CHECKING:
    from nebula.core.situationalawareness.discovery.federationconnector import FederationConnector

SimilarityMetricType = Callable[[OrderedDict, OrderedDict, bool], float | None]
MappingSimilarityType = Callable[[float, float], Annotated[float, "Value in (0, 1]"]]

MAX_HISTORIC_SIZE = 10  # Number of historic data storaged
GLOBAL_PRIORITY = 0.8  # Parameter to priorize global vs local metrics
EPSILON = 0.001
SIGMOID_THRESHOLD = 0.92
TOLERANCE_THRESHOLD = 2  # Threshold to start appliying full penalty
SMOOTH_FACTOR = 0.5
# Maybe it should change according to number of updates received
MOMENTUM_ATENUATION_FACTOR = 0.85  # Proportion of past momentums
MOMENTUM_LEARNING_STEP = 1


class Momentum:
    def __init__(
        self,
        node_manager: "FederationConnector",
        nodes,
        dispersion_penalty=True,
        global_priority=GLOBAL_PRIORITY,
        similarity_metric: SimilarityMetricType = cosine_metric,
        mapping_similarity: MappingSimilarityType = lambda sim_value, e=EPSILON: e + ((sim_value + 1) / 2),
    ):
        logging.info("🌐  Initializing Momemtum strategy")
        self._node_manager = node_manager
        self._momentum_historic = {node_id: deque(maxlen=MAX_HISTORIC_SIZE) for node_id in nodes}
        self._momentum_historic_lock = Locker(name="_momentum_historic_lock", async_lock=True)
        self._previous_momentum = 0
        self._similarities_historic = {node_id: deque(maxlen=MAX_HISTORIC_SIZE) for node_id in nodes}
        self._similarities_historic_lock = Locker(name="_similarities_historic_lock", async_lock=True)
        self._model_similarity_metric_lock = Locker(name="_model_similarity_metric_lock", async_lock=True)
        self._model_similarity_metric = similarity_metric
        self._mapping_similarity_func = mapping_similarity
        self._global_prio = global_priority
        self._dispersion_penalty = dispersion_penalty
        self._addr = self._node_manager.engine.addr

    @property
    def nm(self):
        return self._node_manager

    @property
    def msm(self):
        return self._model_similarity_metric

    @property
    def msf(self):
        return self._mapping_similarity_func

    async def _add_similarity_to_node(self, node_id, sim_value):
        # logging.info(f"Adding | node ID: {node_id}, cossine similarity value: {sim_value}")
        self._similarities_historic_lock.acquire_async()
        self._similarities_historic[node_id].append(sim_value)
        self._similarities_historic_lock.release_async()

    async def _get_similarity_historic(self, addrs):
        """
            Get historic storaged for node IDs on 'addrs'

        Args:
            addrs (List)): List of node IDs that has sent update this round
        """
        self._similarities_historic_lock.acquire_async()
        historic = {}
        for key, value in self._similarities_historic.items():
            if key in addrs:
                historic[key] = value
        self._similarities_historic_lock.release_async()
        return historic

    async def update_node(self, node_id, remove=False):
        self._similarities_historic_lock.acquire_async()
        self._momentum_historic_lock.acquire_async()
        logging.info(f"Update | addr: {node_id}, remove: {remove}")
        if remove:
            self._similarities_historic.pop(node_id, None)
        else:
            self._similarities_historic.update({node_id: deque(maxlen=MAX_HISTORIC_SIZE)})
            self._momentum_historic.update({node_id: deque(maxlen=MAX_HISTORIC_SIZE)})
        self._momentum_historic_lock.release_async()
        self._similarities_historic_lock.release_async()

    async def change_similarity_metric(self, new_metric: SimilarityMetricType, new_mapping: MappingSimilarityType):
        self._model_similarity_metric_lock.acquire_async()
        self.msm = new_metric
        self.msf = new_mapping
        # maybe we should remove historic data due to incongruous data
        self._model_similarity_metric_lock.release_async()

    async def _calculate_similarities(self, updates: dict):
        """
            Function to calculate similarity between local model and models received
            using metric function. The value is storaged on the historic

        Args:
            updates (dict): {node ID: model}
        """
        logging.info("Calculate | Model Similarity values are being calculated...")
        model = self.nm.engine.trainer.get_model_parameters()
        for addr, update in updates.items():
            if addr == self._addr:
                continue
            updt_model, _ = update
            sim_value = self.msm(
                model,
                updt_model,
                similarity=True,
            )
            logging.info(f"Model similarity for node: {addr}, sim: {sim_value:.4f}")
            await self._add_similarity_to_node(addr, sim_value)

    def _calculate_dispersion_penalty(self, historic: dict, updates: dict):
        from math import sqrt

        logging.info("Calculate | Dispersion penalty")
        round_similarities = [(addr, n_hist[-1]) for addr, n_hist in historic.items() if n_hist]
        if round_similarities:
            mean_similarity = np.mean(round_similarities)
            std_similarity = np.std(round_similarities)
            n_updates = len(updates) - 1
            logging.info(f"Calculate | mean similarity: {mean_similarity}, standar deviation: {std_similarity}")
            for addr, sim in round_similarities:
                if abs(sim - mean_similarity) < TOLERANCE_THRESHOLD * std_similarity:
                    logging.info(f"Penalty | Dispersion is lower than threshold, for node: {addr}")
                    penalty = (SMOOTH_FACTOR * (abs(sim - mean_similarity) / (std_similarity + EPSILON))) * (
                        1 / sqrt(n_updates)
                    )
                else:
                    logging.info(f"Penalty | Dispersion is higher than threshold, for node: {addr}")
                    penalty = (abs(sim - mean_similarity) / (std_similarity + EPSILON)) * (1 / sqrt(n_updates))

                penalty = min(1.0, max(0.0, penalty))
                logging.info(f"Penalty value: {penalty}")
                dispersion_penalty = 1 - penalty

    def map_value(sim_value, e=EPSILON):
        return e + ((sim_value + 1) / 2)

    async def calculate_momentum_weights(self, updates: dict):
        if len(updates) == 1:
            return
        logging.info("Calculate | Momemtum weights are being calculated...")
        self._model_similarity_metric_lock.acquire_async()
        await self._calculate_similarities(
            updates
        )  # Calculate similarity value between self model and updates received
        historic = await self._get_similarity_historic(
            updates.keys()
        )  # Get historic similarities values from nodes that has sent update this round

        def sigmoid(similarity, k=2.5):
            if similarity >= SIGMOID_THRESHOLD:  # threshold to consider better updates
                sigmoid = 1
            else:
                sigmoid = 1 / (1 + np.exp(-k * (similarity)))
            return sigmoid

        # Calculate round local momentum for each node
        for node_addr, n_hist in historic.items():
            if not n_hist or node_addr == self._addr:
                continue
            sim_value = n_hist[-1]  # Get last similarity value
            mapped_sim_value = self.msf(sim_value)  # Mapped into [0, 1] interval
            smoothed_value = sigmoid(mapped_sim_value)
            local_round_momentum = smoothed_value * self._global_prio + (1 - self._global_prio) * mapped_sim_value
            self._momentum_historic[node_addr].append(local_round_momentum)

        # Calculate round neighborhood momentum
        round_neighborhood_momentum = np.mean([
            self._momentum_historic[node_addr][-1] for node_addr, _ in historic.items()
        ])
        if not self._previous_momentum:
            self._previous_momentum = round_neighborhood_momentum
        else:
            self._previous_momentum = (
                MOMENTUM_ATENUATION_FACTOR * self._previous_momentum
                + (1 - MOMENTUM_ATENUATION_FACTOR) * round_neighborhood_momentum
            )

        for node_addr in historic.keys():
            model, weight = updates[node_addr]
            adjusted_weight = self._previous_momentum * weight  # Aplicar momentum como factor de ajuste

            # updates[node_addr] = (model, adjusted_weight)

            logging.info(
                f"Node {node_addr}: sim={sim_value:.3f}, momentum_vec={self._previous_momentum:.3f}, adjusted_weight={adjusted_weight:.3f}"
            )

        self._model_similarity_metric_lock.release_async()
