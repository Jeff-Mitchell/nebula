import torch
import torch.nn.functional as F
from nebula.core.aggregation.fedavg import FedAvg

class ProserProtoAggregator(FedAvg):
    """
    Agregador específico para ProserProto que extiende FedAvg.
    Agrega los pesos del modelo usando FedAvg y los prototipos usando una estrategia personalizada.
    """
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):
        # Separar los prototipos del resto de los parámetros
        models_list = list(models.values())
        models_params = [m for m, _ in models_list]

        # Extraer y remover temporalmente los prototipos
        prototypes_list = []
        unknown_protos_list = []
        for model_params in models_params:
            if 'prototypes' in model_params:
                protos = model_params.pop('prototypes')
                # Convertir tensores a listas si es necesario
                if isinstance(protos, dict):
                    protos = {k: list(v) if isinstance(v, torch.Tensor) and v.numel() > 0 else []
                            for k, v in protos.items()}
                prototypes_list.append(protos)
            if 'unknown_protos' in model_params:
                unknown = model_params.pop('unknown_protos')
                # Convertir tensores a listas si es necesario
                if isinstance(unknown, dict):
                    unknown = {k: list(v) if isinstance(v, torch.Tensor) and v.numel() > 0 else []
                             for k, v in unknown.items()}
                unknown_protos_list.append(unknown)

        # Agregar los pesos del modelo usando FedAvg
        aggregated_weights = super().run_aggregation(models)

        # Recolectar prototipos
        combined = {}  # { label: [lista de tensores de todos los modelos] }
        combined_unknown = {}

        for protos in prototypes_list:
            # Procesar prototipos conocidos
            for lbl, protolist in protos.items():
                if not protolist:  # Skip empty lists
                    continue
                if lbl not in combined:
                    combined[lbl] = []
                # Asegurar que los prototipos están normalizados
                for proto in protolist:
                    if not isinstance(proto, torch.Tensor):
                        continue
                    if proto.numel() == 0:  # Skip empty tensors
                        continue
                    if proto.dim() == 0:  # Skip scalars
                        continue
                    # Aplanar y normalizar
                    proto = proto.view(-1)
                    proto = F.normalize(proto, p=2, dim=0)
                    combined[lbl].append(proto)

        for protos in unknown_protos_list:
            # Procesar prototipos desconocidos
            for lbl, protolist in protos.items():
                if not protolist:  # Skip empty lists
                    continue
                if lbl not in combined_unknown:
                    combined_unknown[lbl] = []
                # Asegurar que los prototipos están normalizados
                for proto in protolist:
                    if not isinstance(proto, torch.Tensor):
                        continue
                    if proto.numel() == 0:  # Skip empty tensors
                        continue
                    if proto.dim() == 0:  # Skip scalars
                        continue
                    # Aplanar y normalizar
                    proto = proto.view(-1)
                    proto = F.normalize(proto, p=2, dim=0)
                    combined_unknown[lbl].append(proto)

        # Agregación de prototipos conocidos
        global_prototypes = {}
        for lbl, proto_list in combined.items():
            if proto_list:
                try:
                    # Convertir a tensores y promediar
                    protos = torch.stack(proto_list)  # (num_protos, embedding_dim)
                    mean_proto = protos.mean(dim=0)  # (embedding_dim,)
                    # Normalizar el prototipo resultante
                    mean_proto = F.normalize(mean_proto, p=2, dim=0)
                    # Convertir a tensor para mantener consistencia con state_dict
                    global_prototypes[lbl] = mean_proto.unsqueeze(0)
                except:
                    # Si hay error al stackear, ignorar esta clase
                    continue

        # Agregación de prototipos desconocidos
        global_unknown_prototypes = {}
        for lbl, proto_list in combined_unknown.items():
            if proto_list:
                try:
                    # Convertir a tensores y promediar
                    protos = torch.stack(proto_list)  # (num_protos, embedding_dim)
                    mean_proto = protos.mean(dim=0)  # (embedding_dim,)
                    # Normalizar el prototipo resultante
                    mean_proto = F.normalize(mean_proto, p=2, dim=0)
                    # Convertir a tensor para mantener consistencia con state_dict
                    global_unknown_prototypes[lbl] = mean_proto.unsqueeze(0)
                except:
                    # Si hay error al stackear, ignorar esta clase
                    continue

        # Añadir los prototipos agregados al diccionario de pesos
        aggregated_weights['prototypes'] = global_prototypes
        aggregated_weights['unknown_protos'] = global_unknown_prototypes

        return aggregated_weights
