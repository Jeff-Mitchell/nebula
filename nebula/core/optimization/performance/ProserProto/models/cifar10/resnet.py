import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from nebula.core.optimization.performance.ProserProto.models.proserproto_base import ProserProtoNebulaModel

class ProserProtoResNet18(ProserProtoNebulaModel):
    """
    ResNet18 para CIFAR10 que implementa el modelo ProserProto.
    """
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        embedding_dim=256
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            seed=seed,
            embedding_dim=embedding_dim
        )

        # Cargar ResNet18 y quitar la capa final
        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Quitar FC, queda (B, 512, 1, 1)

        # Embedding final
        self.embedding_layer = nn.Linear(512, embedding_dim)

        # Inicialización de pesos mejorada
        nn.init.kaiming_normal_(self.embedding_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.embedding_layer.bias)

        # Criterio de pérdida
        self.criterion = nn.CrossEntropyLoss()

    def get_embedding(self, x):
        """
        Obtiene embeddings para un batch de imágenes.
        """
        x = self.feature_extractor(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)        # (B, 512)
        emb = self.embedding_layer(x)  # (B, embedding_dim)

        # Debug: verificar forma del embedding
        if emb.dim() != 2:
            emb = emb.view(emb.size(0), -1)

        # Normalizar embeddings para comparaciones de distancia más estables
        emb = F.normalize(emb, p=2, dim=1)

        return emb

    def forward(self, x):
        """
        Forward pass que retorna logits para clasificación.
        """
        emb = self.get_embedding(x)
        return self.classify_embeddings(emb)
