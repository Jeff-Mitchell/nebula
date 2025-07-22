import gc
import logging
import torch
from nebula.core.aggregation.aggregator import Aggregator


class ProtoAvg(Aggregator):
    """
    Prototype Aggregation (FedProto) [Yue Tan et al., 2022]
    Paper: https://arxiv.org/abs/2105.00243
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, prototypes):
        try:
            super().run_aggregation(prototypes)

            # Convertir a una lista para poder iterar
            prototypes_list = list(prototypes.values())

            # Filtrar prototipos vacíos y extraer los prototipos y pesos
            non_empty_prototypes = []
            for model_tuple in prototypes_list:
                protos, weight = model_tuple  # Desempaquetar la tupla (protos, weight)
                if isinstance(protos, dict) and len(protos) > 0:
                    non_empty_prototypes.append((protos, weight))
                else:
                    # Opcional: registrar que se omitió un prototipo vacío
                    logging.debug("Se omitió un prototipo vacío.")

            if not non_empty_prototypes:
                # Manejar el caso en que todos los prototipos están vacíos
                logging.warning("Todos los prototipos están vacíos; no se puede realizar la agregación.")
                return dict()

            # Obtener el dispositivo del primer prototipo no vacío
            first_protos = non_empty_prototypes[0][0]  # El primer elemento de la tupla es el diccionario de prototipos
            first_label = next(iter(first_protos))
            device = first_protos[first_label].device

            # Pre-procesar los pesos para evitar operaciones repetidas
            processed_weights = []
            for protos, weight in non_empty_prototypes:
                if isinstance(weight, torch.Tensor):
                    weight = weight.to(device).item()
                else:
                    weight = float(weight)
                processed_weights.append(weight)

            total_samples = sum(processed_weights)
            if total_samples == 0:
                logging.warning("La suma total de pesos es 0; no se puede realizar la agregación.")
                return dict()

            # Crear un prototipo acumulado en el dispositivo correcto
            accum = {}

            # Inicializar acumuladores para todos los labels encontrados
            all_labels = set()
            for protos, _ in non_empty_prototypes:
                all_labels.update(protos.keys())

            # Inicializar acumuladores para todos los labels
            for label in all_labels:
                # Encontrar el primer tensor válido para este label
                for protos, _ in non_empty_prototypes:
                    if label in protos and isinstance(protos[label], torch.Tensor):
                        accum[label] = torch.zeros_like(protos[label]).to(device)
                        break

            # Agregar los prototipos ponderados de forma más eficiente
            for protos, weight in zip(non_empty_prototypes, processed_weights):
                protos_dict = protos[0] if isinstance(protos, tuple) else protos
                for label, proto_tensor in protos_dict.items():
                    if isinstance(proto_tensor, torch.Tensor):
                        if label not in accum:
                            accum[label] = torch.zeros_like(proto_tensor).to(device)
                        accum[label].add_(proto_tensor.to(device) * weight)

            # Normalizar 'accum' de forma más eficiente
            for label in accum:
                accum[label].div_(total_samples)

            del prototypes_list, non_empty_prototypes, processed_weights, total_samples
            gc.collect()
            torch.cuda.empty_cache()  # Liberar memoria GPU si está disponible

            return accum

        except Exception as e:
            logging.error(f"Error en ProtoAvg.run_aggregation: {str(e)}")
            logging.error(f"Tipo de prototypes: {type(prototypes)}")
            if isinstance(prototypes, dict):
                for k, v in prototypes.items():
                    logging.error(f"Clave: {k}, Tipo de valor: {type(v)}")
            # En caso de error, devolver un diccionario vacío para permitir que el sistema continúe
            return dict()
