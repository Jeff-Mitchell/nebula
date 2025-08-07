import torch
import torch.nn as nn
import torch.nn.functional as F

from nebula.core.research.Proser.models.proser_base import ProserNebulaModel


class ProserResNet18CIFAR10Nebula(ProserNebulaModel):
    """
    Modelo PROSER basado en ResNet18 para CIFAR-10 y adaptado a NEBULA.
    Incluye:
    - Arquitectura ResNet18 optimizada para CIFAR-10
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
        embedding_dim=512,
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

        # Build ResNet18 architecture for CIFAR-10
        self._build_resnet18(input_channels)

        # Clasificador con clase dummy añadida
        # self.num_classes ya incluye la dummy class (viene de la clase base)
        # Por tanto, no necesitamos añadir más dummy classes
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        nn.init.zeros_(self.classifier.bias)

    def _build_resnet18(self, input_channels):
        """
        Construye la arquitectura ResNet18 optimizada para CIFAR-10.
        Separa en capas pre y post para permitir manifold mixup.
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

        # Capas ResNet - dividimos en pre (layer1, layer2) y post (layer3, layer4)
        inplanes = 64
        self.layer1, inplanes = make_layer(inplanes, 64, 2)
        self.layer2, inplanes = make_layer(inplanes, 128, 2, stride=2)

        # Punto de separación para manifold mixup
        self.layer3, inplanes = make_layer(inplanes, 256, 2, stride=2)
        self.layer4, inplanes = make_layer(inplanes, 512, 2, stride=2)

        # Capas finales
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embedding = nn.Linear(512 * BasicBlock.expansion, self.embedding_dim)

    def forward(self, x):
        """Forward pass que retorna logits con clase dummy."""
        emb = self.get_embedding(x)
        return self.get_logits_with_dummy(emb)

    def get_embedding(self, x):
        """
        Extrae el embedding completo pasando por todas las capas.
        """
        # Capas iniciales
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Capas pre
        x = self.layer1(x)
        x = self.layer2(x)

        # Capas post
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling y embedding final
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc_embedding(x)

    def get_pre_features(self, x):
        """
        Extrae características antes del punto de mixup (hasta layer2).
        Usado para manifold mixup.
        """
        # Capas iniciales
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Capas pre (hasta layer2)
        x = self.layer1(x)
        x = self.layer2(x)

        return x

    def get_post_features(self, x_pre):
        """
        Proyecta el resultado del mixup a través de las capas finales.
        """
        # Capas post (desde layer3)
        x = self.layer3(x_pre)
        x = self.layer4(x)

        # Pooling y embedding final
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc_embedding(x)

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
