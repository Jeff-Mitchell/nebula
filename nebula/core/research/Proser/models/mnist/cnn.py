import torch
import torch.nn as nn
import torch.nn.functional as F

from nebula.core.research.Proser.models.proser_base import ProserNebulaModel

class ProserCNNNebula(ProserNebulaModel):
    """
    CNN sencilla compatible con ProserNebulaModel.
    Pensada para MNIST / FashionMNIST con clase dummy y manifold mixup.
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
        gamma=0.1,
        embedding_dim=128
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            seed=seed,
            beta=beta,
            gamma=gamma,
        )

        self.embedding_dim = embedding_dim
        self.num_dummy = 1
        self.calibration_bias = 0.0

        # CNN básica
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # Post-convolucional
        # Entrada: 28x28 → pool → 14x14 → pool → 7x7 → sin pool → 64x7x7
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)
        # self.num_classes ya incluye la dummy class (viene de la clase base)
        # Por tanto, no necesitamos añadir más dummy classes
        self.classifier = nn.Linear(embedding_dim, self.num_classes)

        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        emb = self.get_embedding(x)
        return self.get_logits_with_dummy(emb)

    def get_embedding(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # (B, 32, 14, 14)

        x = self.relu(self.conv2(x))
        x = self.pool(x)  # (B, 64, 7, 7)

        x = self.relu(self.conv3(x))  # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)     # (B, 3136)
        return self.fc(x)             # (B, embedding_dim)

    def get_pre_features(self, x):
        """
        Extrae características antes del proyector final (usado para mixup).
        """
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))  # Shape: (B, 64, 7, 7)
        return x

    def get_post_features(self, x_pre):
        """
        Proyecta el resultado del mixup a través de la capa final.
        """
        x = x_pre.view(x_pre.size(0), -1)
        return self.fc(x)

    def get_logits_with_dummy(self, emb):
        logits = self.classifier(emb)
        real_num_classes = self.num_classes - self.num_dummy
        known_logits = logits[:, :real_num_classes]
        dummy_logits = logits[:, real_num_classes:]

        # When num_dummy = 1, we don't need to take max
        if self.num_dummy == 1:
            max_dummy_logits = dummy_logits
        else:
            max_dummy_logits, _ = torch.max(dummy_logits, dim=1, keepdim=True)

        max_dummy_logits = max_dummy_logits - self.calibration_bias

        return torch.cat([known_logits, max_dummy_logits], dim=1)

    def calibrate(self, datamodule, device, target_known_ratio=0.95):
        """
        Calibra el umbral de la clase dummy usando el conjunto de validación.
        Args:
            datamodule: DataModule con los datos de validación
            device: Dispositivo donde realizar los cálculos
            target_known_ratio: Ratio objetivo de muestras clasificadas como conocidas
        Returns:
            float: Valor de calibración calculado
        """
        self.eval()
        # Guardar el dispositivo original del modelo
        original_device = next(self.parameters()).device
        # Mover el modelo al dispositivo correcto
        self.to(device)
        known_maxes, dummy_maxes, total_samples = [], [], 0

        # Aseguramos que el datamodule está configurado
        datamodule.setup('fit')
        val_loader = datamodule.val_dataloader()

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                features = self.get_embedding(x)
                logits = self.classifier(features)

                # Verificar dimensiones para debug
                if logits.size(1) < self.num_classes:
                    raise RuntimeError(f"Error: logits tiene {logits.size(1)} columnas, "
                                     f"pero self.num_classes={self.num_classes}. "
                                     f"Inconsistencia en la arquitectura.")

                # El problema está aquí: self.num_classes ya incluye la dummy class
                # Necesitamos usar self.num_classes - self.num_dummy para las clases reales
                real_num_classes = self.num_classes - self.num_dummy
                known_logits = logits[:, :real_num_classes]
                dummy_logits = logits[:, real_num_classes:]

                known_maxes.extend(known_logits.max(dim=1)[0].cpu().tolist())
                dummy_maxes.extend(dummy_logits.max(dim=1)[0].cpu().tolist())
                total_samples += x.size(0)

        known_maxes = torch.tensor(known_maxes)
        dummy_maxes = torch.tensor(dummy_maxes)
        diffs = dummy_maxes - known_maxes

        sorted_diffs = torch.sort(diffs, descending=True)[0]
        idx = int(len(sorted_diffs) * (1 - target_known_ratio))
        self.calibration_bias = float(sorted_diffs[idx])

        # Validación rápida
        known_preds = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                preds = self.predict(x)
                known_preds += sum(1 for p in preds if p != "unknown")
        actual_ratio = known_preds / total_samples

        # Restaurar el modelo a su dispositivo original
        self.to(original_device)

        return self.calibration_bias

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            features = self.get_embedding(x)
            logits = self.get_logits_with_dummy(features)
            preds = torch.argmax(logits, dim=1)

            real_num_classes = self.num_classes - self.num_dummy
            results = []
            for p in preds:
                if p == real_num_classes:
                    results.append("unknown")
                else:
                    results.append(int(p))
            return results
