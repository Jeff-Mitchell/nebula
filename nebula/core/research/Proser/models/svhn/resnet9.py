import torch
import torch.nn as nn
import torch.nn.functional as F

from nebula.core.research.Proser.models.proser_base import ProserNebulaModel


class ProserResNet9SVHNNebula(ProserNebulaModel):
    """
    Modelo PROSER basado en ResNet9 para SVHN y adaptado a NEBULA.
    Incluye:
    - Arquitectura ResNet9 optimizada para SVHN (32x32x3)
    - Extracción de embeddings con separación pre/post para mixup
    - Clasificador extendido con clase dummy
    - Calibración de umbral para separación conocida/desconocida
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        beta=1.0,
        gamma=0.1,
        embedding_dim=128,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,  # Se añade clase dummy internamente
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

        # Build ResNet9 architecture for SVHN
        self._build_resnet9(input_channels)

        # Clasificador con clase dummy añadida
        # self.num_classes ya incluye la dummy class (viene de la clase base)
        # Por tanto, no necesitamos añadir más dummy classes
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        nn.init.zeros_(self.classifier.bias)

    def _build_resnet9(self, input_channels):
        """
        Construye la arquitectura ResNet9 optimizada para SVHN.
        Separa en capas pre y post para permitir manifold mixup.

        Args:
            input_channels (int): Número de canales de entrada (3 para SVHN)
        """

        def conv_block(input_channels, output_channels, pool=False):
            """
            Bloque convolucional básico: Conv2d + BatchNorm + ReLU + (Optional MaxPool)
            """
            layers = [
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        # Capas iniciales (pre-mixup)
        self.conv1 = conv_block(input_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)

        # Primer bloque residual (pre-mixup)
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )

        # Punto de separación para manifold mixup

        # Segunda secuencia convolucional (post-mixup)
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)

        # Segundo bloque residual (post-mixup)
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )

        # Pooling y embedding final
        self.final_pool = nn.MaxPool2d(4)
        self.fc_embedding = nn.Linear(512, self.embedding_dim)

        # Inicialización de pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass que retorna logits con clase dummy."""
        emb = self.get_embedding(x)
        return self.get_logits_with_dummy(emb)

    def get_embedding(self, x):
        """
        Extrae el embedding completo pasando por todas las capas.
        """
        # Capas pre (hasta res1)
        out = self.conv1(x)
        out = self.conv2(out)
        identity1 = out
        out = self.res1(out) + identity1  # Conexión residual

        # Capas post (desde conv3)
        out = self.conv3(out)
        out = self.conv4(out)
        identity2 = out
        out = self.res2(out) + identity2  # Conexión residual

        # Pooling y embedding final
        out = self.final_pool(out)
        out = torch.flatten(out, 1)
        return self.fc_embedding(out)

    def get_pre_features(self, x):
        """
        Extrae características antes del punto de mixup (hasta res1).
        Usado para manifold mixup.
        """
        # Capas pre (hasta res1)
        out = self.conv1(x)
        out = self.conv2(out)
        identity1 = out
        out = self.res1(out) + identity1  # Conexión residual

        return out

    def get_post_features(self, x_pre):
        """
        Proyecta el resultado del mixup a través de las capas finales.
        """
        # Capas post (desde conv3)
        out = self.conv3(x_pre)
        out = self.conv4(out)
        identity2 = out
        out = self.res2(out) + identity2  # Conexión residual

        # Pooling y embedding final
        out = self.final_pool(out)
        out = torch.flatten(out, 1)
        return self.fc_embedding(out)

    def get_logits_with_dummy(self, emb):
        """
        Aplica clasificador con clase dummy y calibración.
        """
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

                # self.num_classes ya incluye la dummy class
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
        """
        Realiza predicción clasificando muestras como conocidas o desconocidas.
        """
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
