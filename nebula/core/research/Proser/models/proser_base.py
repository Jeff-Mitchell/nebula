import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from nebula.core.models.nebulamodel import NebulaModel
class ProserNebulaModel(NebulaModel):
    """
    Modelo PROSER adaptado al ecosistema NEBULA.
    Implementa:
      - Pérdida de clasificación estándar con clase dummy.
      - Placeholder de datos mediante manifold mixup.
    """

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        beta=1.0,
        gamma=0.1
    ):
        super().__init__(input_channels, num_classes + 1, learning_rate, metrics, confusion_matrix, seed)

        self.beta = beta    # Peso para la pérdida dummy
        self.gamma = gamma  # Peso para la pérdida mixup
        self.criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = None

    def get_embedding(self, x):
        """
        Método abstracto a implementar por subclases.
        Devuelve los embeddings antes de la capa de clasificación.
        """
        raise NotImplementedError

    def get_logits_with_dummy(self, emb):
        """
        Devuelve logits incluyendo clase dummy.
        """
        raise NotImplementedError

    def get_pre_features(self, x):
        """
        Extrae características previas a la capa de mezcla.
        """
        raise NotImplementedError

    def get_post_features(self, features):
        """
        Proyecta las características tras mixup hacia el espacio final.
        """
        raise NotImplementedError

    def calibrate(self, datamodule, device, target_known_ratio=0.95):
        """
        Calibración de la clase dummy usando el datamodule.
        Args:
            datamodule: DataModule con los datos de validación
            device: Dispositivo donde realizar los cálculos
            target_known_ratio: Ratio objetivo de muestras clasificadas como conocidas
        Returns:
            float: Valor de calibración calculado
        """
        raise NotImplementedError

    def _compute_classifier_placeholder_loss(self, x, y):
        emb = self.get_embedding(x)
        logits = self.get_logits_with_dummy(emb)

        loss_cls = self.criterion(logits, y)

        # Enmascarar logits reales para dummy loss
        batch_size = logits.size(0)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), y] = False
        logits_masked = logits.clone()
        logits_masked[~mask] = float('-inf')

        dummy_targets = torch.full((batch_size,), logits.size(1)-1, device=x.device)
        loss_dummy = self.criterion(logits_masked, dummy_targets)

        return loss_cls, self.beta * loss_dummy, logits

    def _compute_data_placeholder_loss(self, x1, x2):
        pre_features1 = self.get_pre_features(x1)
        pre_features2 = self.get_pre_features(x2)

        lam = np.random.beta(0.2, 0.2)
        lam = torch.tensor(lam, dtype=torch.float32, device=x1.device)
        lam = lam.view(1, 1)

        mixed = lam * pre_features1 + (1 - lam) * pre_features2

        post_features = self.get_post_features(mixed)
        logits = self.get_logits_with_dummy(post_features)

        dummy_targets = torch.full((logits.size(0),), logits.size(1)-1, device=x1.device)
        loss_mixup = self.criterion(logits, dummy_targets)

        return self.gamma * loss_mixup

    def training_step(self, batch, batch_idx):
        x, y = batch
        if x.size(0) < 2:
            return None

        half = x.size(0) // 2
        x1, y1 = x[:half], y[:half]
        x2 = x[half:2*half]

        if x1.size(0) == 0 or x2.size(0) == 0 or x1.size(0) != x2.size(0):
            return None

        loss_cls, loss_dummy, logits = self._compute_classifier_placeholder_loss(x1, y1)
        loss_mixup = self._compute_data_placeholder_loss(x1, x2)

        loss_total = loss_cls + loss_dummy + loss_mixup

        losses = {
            "Loss_Classification": loss_cls,
            "Loss_Dummy": loss_dummy,
            "Loss_Mixup": loss_mixup,
            "Loss_Total": loss_total
        }

        self.process_metrics("Train", logits, y1, losses)
        return loss_total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        emb = self.get_embedding(x)
        logits = self.get_logits_with_dummy(emb)
        loss = self.criterion(logits, y)
        self.process_metrics("Validation", logits, y, loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        emb = self.get_embedding(x)
        logits = self.get_logits_with_dummy(emb)
        loss = self.criterion(logits, y)
        y_pred_classes = torch.argmax(logits, dim=1)
        accuracy = torch.mean((y_pred_classes == y).float())

        if dataloader_idx == 0:
            self.log("val_loss", loss, on_epoch=True)
            self.log("val_accuracy", accuracy, on_epoch=True)
            return self.step(batch, batch_idx, phase="Test (Local)")
        else:
            return self.step(batch, batch_idx, phase="Test (Global)")

    def configure_optimizers(self):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self._optimizer

    def process_metrics(self, phase, y_pred, y, loss=None):
        y_pred_classes = torch.argmax(y_pred, dim=1).detach()
        y = y.detach()

        if phase == "Train":
            if isinstance(loss, dict):
                for name, val in loss.items():
                    self.log(f"{phase}/{name}", val.detach(), on_step=True, on_epoch=True, prog_bar=True)
            elif loss is not None:
                self.log(f"{phase}/Loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
            self.train_metrics.update(y_pred_classes, y)

        elif phase == "Validation":
            self.val_metrics.update(y_pred_classes, y)
        elif phase == "Test (Local)":
            self.test_metrics.update(y_pred_classes, y)
            if self.cm:
                self.cm.update(y_pred_classes, y)
        elif phase == "Test (Global)":
            self.test_metrics_global.update(y_pred_classes, y)
            if self.cm_global:
                self.cm_global.update(y_pred_classes, y)
        else:
            raise NotImplementedError

        del y_pred_classes, y

    def on_train_epoch_end(self):
        self.log_metrics_end("Train")
        self.train_metrics.reset()
        self.global_number["Train"] += 1
