import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nebula.core.optimization.performance.ProserProto.models.proserproto_base import ProserProtoNebulaModel


class ProserProtoResNet18CIFAR10(ProserProtoNebulaModel):
    """
    Modelo ProserProto basado en ResNet18 para CIFAR-10.
    Combina la arquitectura ResNet18 con el algoritmo ProserProto que incluye:
    - Manifold mixup entre embeddings conocidos y desconocidos
    - Manejo de prototipos para clasificación
    - Pérdidas de diversidad y separación de prototipos
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        embedding_dim=512,
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

        # Build ResNet18 architecture for CIFAR-10
        self._build_resnet18(input_channels)

    def _build_resnet18(self, input_channels):
        """
        Construye la arquitectura ResNet18 optimizada para CIFAR-10.
        """

        def conv3x3(in_planes, out_planes, stride=1):
            """3x3 convolution with padding"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)

        def conv1x1(in_planes, out_planes, stride=1):
            """1x1 convolution"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

        def make_layer(inplanes, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or inplanes != planes * BasicBlock.expansion:
                downsample = nn.Sequential(
                    conv1x1(inplanes, planes * BasicBlock.expansion, stride),
                    nn.BatchNorm2d(planes * BasicBlock.expansion),
                )

            layers = []
            layers.append(BasicBlock(inplanes, planes, stride, downsample))
            inplanes = planes * BasicBlock.expansion
            for _ in range(1, blocks):
                layers.append(BasicBlock(inplanes, planes))

            return nn.Sequential(*layers), inplanes

        # Capas iniciales
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Capas ResNet
        inplanes = 64
        self.layer1, inplanes = make_layer(inplanes, 64, 2)
        self.layer2, inplanes = make_layer(inplanes, 128, 2, stride=2)
        self.layer3, inplanes = make_layer(inplanes, 256, 2, stride=2)
        self.layer4, inplanes = make_layer(inplanes, 512, 2, stride=2)

        # Capas finales
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embedding = nn.Linear(512 * BasicBlock.expansion, self.embedding_dim)

        # Inicialización de pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_embedding(self, x):
        """
        Extrae el embedding completo pasando por todas las capas ResNet18.

        Args:
            x (torch.Tensor): Input tensor de shape (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: Embedding de shape (batch_size, embedding_dim)
        """
        # Capas iniciales
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Capas ResNet
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling y embedding final
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc_embedding(x)

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
