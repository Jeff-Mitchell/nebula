import torch
import torch.nn as nn
import torch.nn.functional as F

from nebula.core.research.FedProto.models.fedprotonebulamodel import FedProtoNebulaModel


class FedProtoSVHNModelResNet9(FedProtoNebulaModel):
    """
    FedProto ResNet9 model para SVHN dataset.

    Implementa la arquitectura ResNet9 optimizada para SVHN (32x32x3) con el algoritmo FedProto
    que incluye aprendizaje basado en prototipos y agregación federada.

    La arquitectura ResNet9 incluye:
    - 2 bloques convolucionales básicos
    - 2 bloques residuales
    - Conexiones skip para mejor flujo de gradientes
    - Clasificador final con manejo de prototipos
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        beta=1,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, beta)

        self.example_input_array = torch.rand(1, 3, 32, 32)

        # Build ResNet9 architecture
        self._build_resnet9(input_channels, num_classes)

    def _build_resnet9(self, input_channels, num_classes):
        """
        Construye la arquitectura ResNet9 optimizada para SVHN.

        Args:
            input_channels (int): Número de canales de entrada (3 para SVHN)
            num_classes (int): Número de clases de salida (10 para SVHN)
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

        # Clasificador final
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 128),  # Embedding layer for prototypes
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward_train(self, x, softmax=True, is_feat=False):
        """
        Forward pass para entrenamiento con FedProto.

        Args:
            x (torch.Tensor): Input tensor de shape (batch_size, 3, 32, 32)
            softmax (bool): Si aplicar log_softmax a los logits
            is_feat (bool): Si retornar features intermedias

        Returns:
            tuple: (logits, prototypes, [features]) si is_feat=True
                   (logits, prototypes) si is_feat=False
        """
        # Forward pass a través de ResNet9
        out = self.conv1(x)

        out = self.conv2(out)
        identity1 = out
        out = self.res1(out) + identity1  # Conexión residual

        conv3_out = self.conv3(out)
        out = self.conv4(conv3_out)
        identity2 = out
        out = self.res2(out) + identity2  # Conexión residual

        # Clasificador
        pooled = nn.MaxPool2d(4)(out)
        flattened = nn.Flatten()(pooled)

        # Embedding layer (prototypes)
        prototypes = self.classifier[2](flattened)  # Linear(512, 128)
        prototypes = self.classifier[3](prototypes)  # ReLU

        # Final classification
        logits = self.classifier[4](prototypes)  # Linear(128, num_classes)

        # Cleanup intermediate tensors
        del flattened, pooled, identity1, identity2

        if is_feat:
            features = [out, conv3_out]  # Return some intermediate features
            if softmax:
                return F.log_softmax(logits, dim=1), prototypes, features
            return logits, prototypes, features

        del conv3_out

        if softmax:
            return F.log_softmax(logits, dim=1), prototypes
        return logits, prototypes

    def forward(self, x):
        """
        Forward pass para inferencia usando prototipos globales si están disponibles.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Logits de clasificación
        """
        if len(self.global_protos) == 0:
            # Si no hay prototipos globales, usar forward normal
            logits, _ = self.forward_train(x, softmax=False)
            return F.log_softmax(logits, dim=1)
        else:
            # Usar prototipos globales para clasificación
            # Forward hasta obtener embeddings
            out = self.conv1(x)

            out = self.conv2(out)
            identity1 = out
            out = self.res1(out) + identity1

            out = self.conv3(out)
            out = self.conv4(out)
            identity2 = out
            out = self.res2(out) + identity2

            # Extraer embeddings
            pooled = nn.MaxPool2d(4)(out)
            flattened = nn.Flatten()(pooled)
            embeddings = self.classifier[2](flattened)  # Linear(512, 128)
            embeddings = self.classifier[3](embeddings)  # ReLU

            # Clasificación basada en prototipos
            logits = []
            for emb in embeddings:
                distances = []
                for class_idx in range(self.num_classes):
                    if class_idx in self.global_protos:
                        # Distancia al prototipo global de la clase
                        proto = self.global_protos[class_idx]
                        dist = torch.norm(emb - proto, p=2)
                        distances.append(-dist)  # Negativo para convertir distancia en similitud
                    else:
                        distances.append(torch.tensor(-float('inf'), device=x.device))
                logits.append(torch.stack(distances))

            logits = torch.stack(logits)
            return F.log_softmax(logits, dim=1)

    def configure_optimizers(self):
        """
        Configura el optimizador Adam.

        Returns:
            torch.optim.Optimizer: Optimizador configurado
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
            weight_decay=1e-4
        )
        self._optimizer = optimizer
        return optimizer
