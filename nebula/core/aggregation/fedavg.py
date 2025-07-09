import gc

import torch

from nebula.core.aggregation.aggregator import Aggregator


class FedAvg(Aggregator):
    """
    Aggregator: Federated Averaging (FedAvg)
    Authors: McMahan et al.
    Year: 2016
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):
        super().run_aggregation(models)

        models = list(models.values())
        total_samples = float(sum(weight for _, weight in models))

        if total_samples == 0:
            raise ValueError("Total number of samples must be greater than zero.")

        last_model_params = models[-1][0]
        accum = {}
        non_tensor_params = {}

        # Inicializar acumuladores
        for layer, param in last_model_params.items():
            if isinstance(param, torch.Tensor):
                accum[layer] = torch.zeros_like(param, dtype=torch.float32)
            else:
                # Para valores no-tensor (como diccionarios de prototipos), usar el último valor
                non_tensor_params[layer] = param

        with torch.no_grad():
            for model_parameters, weight in models:
                normalized_weight = weight / total_samples
                for layer, param in model_parameters.items():
                    if isinstance(param, torch.Tensor):
                        if layer not in accum:
                            accum[layer] = torch.zeros_like(param, dtype=torch.float32)
                        accum[layer].add_(
                            param.to(accum[layer].dtype),
                            alpha=normalized_weight,
                        )
                    else:
                        # Para valores no-tensor, usar el último valor
                        non_tensor_params[layer] = param

        # Combinar parámetros tensor y no-tensor
        accum.update(non_tensor_params)

        del models
        gc.collect()

        return accum
