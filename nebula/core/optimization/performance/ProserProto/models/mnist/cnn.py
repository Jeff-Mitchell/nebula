import torch
import torch.nn as nn
import torch.nn.functional as F
from nebula.core.optimization.performance.ProserProto.models.proserproto_base import ProserProtoNebulaModel

class ProserProtoCNN(ProserProtoNebulaModel):
    """
    CNN para MNIST que implementa el modelo ProserProto.
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
        alpha=0.75
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            seed=seed,
            embedding_dim=embedding_dim,
            alpha=alpha
        )

        # CNN sencilla para MNIST
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        # Embedding final
        # Con 2 poolings, para MNIST 28x28 => (64, 7, 7) => 3136
        self.fc = nn.Linear(64*7*7, embedding_dim)

        # Criterio de pérdida
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass que retorna logits para clasificación.
        """
        emb = self.get_embedding(x)
        return self.classify_embeddings(emb)

    def get_embedding(self, x):
        """
        Obtiene el embedding de la entrada x.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # 32, 14, 14

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)  # 64, 7, 7

        x = self.conv3(x)
        x = self.relu(x)
        # sin pooling final => 64, 7, 7

        x = x.view(x.size(0), -1)  # (B, 64*7*7)
        emb = self.fc(x)
        return emb
