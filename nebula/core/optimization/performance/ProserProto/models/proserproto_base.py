import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.adam import Adam
from nebula.core.models.nebulamodel import NebulaModel

class ProserProtoNebulaModel(NebulaModel):
    """
    Clase base para modelos ProserProto que extiende de NebulaModel.
    Implementa la funcionalidad común para manejar prototipos y embeddings.
    """
    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        embedding_dim=128,
        alpha=0.75,
        lambda_div=1.0,
        lambda_proto=1.0
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.lambda_div = lambda_div
        self.lambda_proto = lambda_proto

        # Diccionarios de prototipos
        self.prototypes = {}
        self.unknown_protos = {}  # p.ej. {'dummy': [list of prototypes for unknown]}
        self.known_classes = []   # solo para referencia

    def state_dict(self):
        """
        Sobrescribe state_dict para incluir los prototipos en el estado del modelo.
        """
        state = {}
        # Obtener los parámetros del modelo
        for name, param in self.named_parameters():
            state[name] = param.data
        # Convertir los prototipos a tensores para serialización
        state['prototypes'] = {k: torch.stack(v) if v else torch.tensor([]) for k, v in self.prototypes.items()}
        state['unknown_protos'] = {k: torch.stack(v) if v else torch.tensor([]) for k, v in self.unknown_protos.items()}
        state['known_classes'] = self.known_classes
        return state

    def load_state_dict(self, state_dict):
        """
        Sobrescribe load_state_dict para cargar los prototipos desde el estado.
        """
        # Extraer los prototipos del estado
        if 'prototypes' in state_dict:
            self.prototypes = {k: list(v) if v.numel() > 0 else [] for k, v in state_dict.pop('prototypes').items()}
        if 'unknown_protos' in state_dict:
            self.unknown_protos = {k: list(v) if v.numel() > 0 else [] for k, v in state_dict.pop('unknown_protos').items()}
        if 'known_classes' in state_dict:
            self.known_classes = state_dict.pop('known_classes')
        # Cargar el resto del estado
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name]

    def manifold_mixup(self, xk, xu):
        """
        Implementa manifold mixup entre embeddings conocidos y desconocidos.
        """
        batch_size = min(xk.size(0), xu.size(0))
        if batch_size < 2:
            return None, None, None

        # Obtener embeddings de ambos conjuntos
        emb_known = self.get_embedding(xk[:batch_size])
        emb_unknown = self.get_embedding(xu[:batch_size])

        # Generar coeficientes de mixup
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = torch.from_numpy(lam).float().to(xk.device)
        lam = lam.view(-1, 1)  # Para broadcasting

        # Realizar mixup - asegurar que los tensores tienen gradientes
        mixed_embeddings = lam * emb_known + (1 - lam) * emb_unknown

        return mixed_embeddings, emb_known, emb_unknown

    def compute_proto_loss(self, emb_known, emb_unknown):
        """
        Calcula pérdida de prototipo para forzar separación entre regiones.
        """
        if emb_known is None or emb_unknown is None:
            return torch.tensor(0.0).to(self.device)

        # Normalizar embeddings
        emb_known = F.normalize(emb_known, p=2, dim=1)
        emb_unknown = F.normalize(emb_unknown, p=2, dim=1)

        # Calcular centroides
        centroid_known = emb_known.mean(dim=0, keepdim=True)
        centroid_unknown = emb_unknown.mean(dim=0, keepdim=True)

        # Pérdida de prototipo: maximizar distancia entre centroides
        proto_loss = -torch.cdist(centroid_known, centroid_unknown).mean()
        return proto_loss

    def compute_diversity_loss(self, mixed_embeddings):
        """
        Calcula pérdida de diversidad para evitar colapso de embeddings.
        """
        if mixed_embeddings is None:
            return torch.tensor(0.0).to(self.device)

        # Normalizar embeddings
        mixed_embeddings = F.normalize(mixed_embeddings, p=2, dim=1)

        # Calcular matriz de similitud coseno
        sim_matrix = torch.mm(mixed_embeddings, mixed_embeddings.t())

        # Máscara para excluir la diagonal
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)

        # Pérdida de diversidad: minimizar similitud entre embeddings diferentes
        diversity_loss = (sim_matrix * mask).mean()
        return diversity_loss

    def training_step(self, batch, batch_idx):
        """
        Paso de entrenamiento que implementa manifold mixup y actualización de prototipos.
        """
        # Desempaquetar batch en formato estándar de NEBULA
        x, y = batch

        # Verificar tamaño mínimo para mixup
        if x.size(0) < 2:
            return None

        # Dividir batch en mitades para simular datos conocidos y desconocidos
        half = x.size(0) // 2
        xk, yk = x[:half], y[:half]  # Datos conocidos
        xu = x[half:]                # Datos desconocidos

        # Forward pass normal para obtener embeddings y logits
        emb_known = self.get_embedding(xk)
        logits = self.classify_embeddings(emb_known)

        # Pérdida de clasificación
        loss_cls = F.cross_entropy(logits, yk)

        # Manifold mixup
        mixed_embeddings, emb_k, emb_u = self.manifold_mixup(xk, xu)

        # Inicializar diccionario de pérdidas
        losses = {
            'Loss_Classification': loss_cls,
            'Loss_Proto': torch.tensor(0.0, device=loss_cls.device),
            'Loss_Diversity': torch.tensor(0.0, device=loss_cls.device)
        }

        # Verificar si el mixup fue exitoso
        if mixed_embeddings is not None:
            # Pérdidas adicionales
            losses['Loss_Proto'] = self.compute_proto_loss(emb_k, emb_u)
            losses['Loss_Diversity'] = self.compute_diversity_loss(mixed_embeddings)

            # Pérdida total
            total_loss = loss_cls + self.lambda_proto * losses['Loss_Proto'] + self.lambda_div * losses['Loss_Diversity']
            losses['Loss_Total'] = total_loss
        else:
            # Si el mixup falló, usar solo pérdida de clasificación
            losses['Loss_Total'] = loss_cls

        # Actualizar prototipos conocidos
        with torch.no_grad():
            for i, label in enumerate(yk):
                self.add_known_prototype(label.item(), emb_known[i].detach())

        # Procesar métricas usando el sistema de NEBULA con todas las pérdidas
        self.process_metrics("Train", logits, yk, losses)

        return losses['Loss_Total']

    def validation_step(self, batch, batch_idx):
        """
        Paso de validación usando los prototipos actuales.
        """
        x, y = batch
        emb = self.get_embedding(x)
        logits = self.classify_embeddings(emb)
        loss = F.cross_entropy(logits, y)

        # Procesar métricas usando el sistema de NEBULA
        self.process_metrics("Validation", logits, y, loss)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Paso de prueba usando los prototipos finales.
        """
        x, y = batch
        emb = self.get_embedding(x)
        logits = self.classify_embeddings(emb)
        loss = F.cross_entropy(logits, y)
        y_pred_classes = torch.argmax(logits, dim=1)
        accuracy = torch.mean((y_pred_classes == y).float())

        # Procesar métricas usando el sistema de NEBULA
        if dataloader_idx == 0:
            self.log(f"val_loss", loss, on_epoch=True, prog_bar=False)
            self.log(f"val_accuracy", accuracy, on_epoch=True, prog_bar=False)
            return self.step(batch, batch_idx=batch_idx, phase="Test (Local)")
        else:
            return self.step(batch, batch_idx=batch_idx, phase="Test (Global)")

        return loss

    def on_train_epoch_end(self):
        """
        Llamado al final de cada época de entrenamiento.
        """
        self.log_metrics_end("Train")
        self.train_metrics.reset()
        self.global_number["Train"] += 1

    def on_validation_epoch_end(self):
        """
        Llamado al final de cada época de validación.
        """
        self.log_metrics_end("Validation")
        self.val_metrics.reset()
        self.global_number["Validation"] += 1

    def on_test_epoch_end(self):
        """
        Llamado al final de cada época de prueba.
        """
        self.log_metrics_end("Test (Local)")
        self.log_metrics_end("Test (Global)")
        self.generate_confusion_matrix("Test (Local)", print_cm=True, plot_cm=True)
        self.generate_confusion_matrix("Test (Global)", print_cm=True, plot_cm=True)
        self.test_metrics.reset()
        self.test_metrics_global.reset()
        self.global_number["Test (Local)"] += 1
        self.global_number["Test (Global)"] += 1

    def configure_optimizers(self):
        """
        Configura el optimizador Adam por defecto.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self._optimizer = optimizer
        return optimizer

    # -----------------------------------------------------------------------------------
    # Métodos abstractos que las subclases deben implementar
    # -----------------------------------------------------------------------------------
    def get_embedding(self, x):
        """
        Obtiene el embedding de la entrada x.
        Debe ser implementado por las subclases.
        """
        raise NotImplementedError

    def classify_embeddings(self, embeddings):
        """
        Clasifica embeddings usando los prototipos actuales.
        Por defecto usa distancia euclidiana.
        """
        logits = []
        for emb in embeddings:
            class_distances = []
            for class_idx in range(self.num_classes):
                if class_idx in self.prototypes and len(self.prototypes[class_idx]) > 0:
                    # Calcular distancia media a los prototipos de la clase
                    protos = torch.stack(self.prototypes[class_idx])
                    dist = torch.mean(torch.cdist(emb.unsqueeze(0), protos))
                    class_distances.append(-dist)  # Negativo para convertir distancia en similitud
                else:
                    class_distances.append(torch.tensor(-float('inf')).to(emb.device))

            logits.append(torch.stack(class_distances))

        return torch.stack(logits)

    # -----------------------------------------------------------------------------------
    # Métodos para actualizar / manejar prototipos
    # -----------------------------------------------------------------------------------
    def add_known_prototype(self, label, emb):
        """
        Agrega un embedding a la lista de prototipos de label.
        """
        if label not in self.prototypes:
            self.prototypes[label] = []
        self.prototypes[label].append(emb.detach())

    def add_unknown_prototype(self, emb):
        """
        Agrega un embedding al prototipo 'dummy' (en self.unknown_protos).
        """
        label_unk = 'dummy'
        if label_unk not in self.unknown_protos:
            self.unknown_protos[label_unk] = []
        self.unknown_protos[label_unk].append(emb.detach())

    def set_prototypes(self, label, emb_list):
        """
        Establece una lista de embeddings como prototipos de label.
        """
        self.prototypes[label] = [e.detach() for e in emb_list]

    def process_metrics(self, phase, y_pred, y, loss=None):
        """
        Calcula y registra métricas para la fase dada.
        Extiende el método base para incluir las pérdidas específicas de ProserProto.
        Args:
            phase (str): Una de 'Train', 'Validation', 'Test (Local)' o 'Test (Global)'
            y_pred (torch.Tensor): Predicciones del modelo
            y (torch.Tensor): Etiquetas verdaderas
            loss (dict or torch.Tensor): Pérdida total o diccionario con componentes de pérdida
        """
        y_pred_classes = torch.argmax(y_pred, dim=1).detach()
        y = y.detach()

        if phase == "Train":
            # Si loss es un diccionario, registrar cada componente
            if isinstance(loss, dict):
                for loss_name, loss_value in loss.items():
                    if loss_value is not None:
                        self.log(f"{phase}/{loss_name}", loss_value.detach(), on_step=True, on_epoch=True, prog_bar=True)
            else:
                if loss is not None:
                    self.log(f"{phase}/Loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

            # Actualizar métricas de clasificación
            self.train_metrics.update(y_pred_classes, y)

        elif phase == "Validation":
            if isinstance(loss, dict):
                for loss_name, loss_value in loss.items():
                    if loss_value is not None:
                        self.log(f"{phase}/{loss_name}", loss_value.detach(), on_epoch=True, prog_bar=True)
            else:
                if loss is not None:
                    self.log(f"{phase}/Loss", loss.detach(), on_epoch=True, prog_bar=True)
            self.val_metrics.update(y_pred_classes, y)

        elif phase == "Test (Local)":
            self.test_metrics.update(y_pred_classes, y)
            self.cm.update(y_pred_classes, y) if self.cm is not None else None

        elif phase == "Test (Global)":
            self.test_metrics_global.update(y_pred_classes, y)
            self.cm_global.update(y_pred_classes, y) if self.cm_global is not None else None

        else:
            raise NotImplementedError

        del y_pred_classes, y
