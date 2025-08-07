import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nebula.core.optimization.performance.ProserProto.models.proserproto_base import ProserProtoNebulaModel


class ProserProtoResNet9SVHN(ProserProtoNebulaModel):
    """
    Modelo ProserProto basado en ResNet9 para SVHN.
    Combina la arquitectura ResNet9 con el algoritmo ProserProto que incluye:
    - Manifold mixup entre embeddings conocidos y desconocidos
    - Manejo de prototipos para clasificación
    - Pérdidas de diversidad y separación de prototipos
    - Arquitectura ResNet9 optimizada para SVHN (32x32x3)
    """

    def __init__(
        self,
        input_channels=3,
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
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            seed=seed,
            embedding_dim=embedding_dim,
            alpha=alpha,
            lambda_div=lambda_div,
            lambda_proto=lambda_proto
        )

        # Build ResNet9 architecture for SVHN
        self._build_resnet9(input_channels)

    def _build_resnet9(self, input_channels):
        """
        Construye la arquitectura ResNet9 optimizada para SVHN.

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

        # Primera secuencia convolucional
        self.conv1 = conv_block(input_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)

        # Primer bloque residual
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )

        # Segunda secuencia convolucional
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)

        # Segundo bloque residual
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

    def get_embedding(self, x):
        """
        Extrae el embedding completo pasando por todas las capas ResNet9.

        Args:
            x (torch.Tensor): Input tensor de shape (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: Embedding de shape (batch_size, embedding_dim)
        """
        # Primera secuencia convolucional
        out = self.conv1(x)

        out = self.conv2(out)
        identity1 = out
        out = self.res1(out) + identity1  # Conexión residual

        # Segunda secuencia convolucional
        out = self.conv3(out)
        out = self.conv4(out)
        identity2 = out
        out = self.res2(out) + identity2  # Conexión residual

        # Pooling y embedding final
        out = self.final_pool(out)
        out = torch.flatten(out, 1)
        return self.fc_embedding(out)

    def classify_embeddings(self, embeddings):
        """
        Clasifica embeddings usando los prototipos actuales.
        Usa distancia euclidiana para determinar la clase más cercana.

        Args:
            embeddings (torch.Tensor): Embeddings a clasificar

        Returns:
            torch.Tensor: Logits de clasificación
        """
        if len(self.prototypes) == 0:
            # Si no hay prototipos, usar clasificación aleatoria
            batch_size = embeddings.size(0)
            return torch.randn(batch_size, self.num_classes, device=embeddings.device)

        logits = []
        for emb in embeddings:
            class_distances = []
            for class_idx in range(self.num_classes):
                if class_idx in self.prototypes and len(self.prototypes[class_idx]) > 0:
                    # Calcular distancia media a los prototipos de la clase
                    protos = torch.stack(self.prototypes[class_idx])
                    # Normalizar embeddings y prototipos para estabilidad
                    emb_norm = F.normalize(emb.unsqueeze(0), p=2, dim=1)
                    protos_norm = F.normalize(protos, p=2, dim=1)

                    # Usar similitud coseno en lugar de distancia euclidiana
                    similarities = torch.mm(emb_norm, protos_norm.t())
                    mean_similarity = similarities.mean()
                    class_distances.append(mean_similarity)
                else:
                    # Si no hay prototipos para esta clase, usar valor muy bajo
                    class_distances.append(torch.tensor(-10.0, device=emb.device))

            logits.append(torch.stack(class_distances))

        return torch.stack(logits)

    def forward(self, x):
        """
        Forward pass estándar para inferencia.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Logits de clasificación
        """
        emb = self.get_embedding(x)
        return self.classify_embeddings(emb)

    def configure_optimizers(self):
        """
        Configura el optimizador Adam con weight decay.

        Returns:
            torch.optim.Optimizer: Optimizador configurado
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self._optimizer = optimizer
        return optimizer
