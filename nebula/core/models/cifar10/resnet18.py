import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from nebula.core.models.nebulamodel import NebulaModel


class CIFAR10ModelResNet18(NebulaModel):
    """
    ResNet18 implementation for CIFAR-10.

    This implementation provides a ResNet18 architecture optimized for CIFAR-10 images (32x32x3).
    Uses residual connections to enable training of deeper networks with better gradient flow.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        if metrics is None:
            metrics = MetricCollection([
                MulticlassAccuracy(num_classes=num_classes),
                MulticlassPrecision(num_classes=num_classes),
                MulticlassRecall(num_classes=num_classes),
                MulticlassF1Score(num_classes=num_classes),
            ])
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        if confusion_matrix is None:
            self.cm = MulticlassConfusionMatrix(num_classes=num_classes)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

        # Build ResNet18 architecture
        self.model = self._build_resnet18(input_channels, num_classes)

    def _build_resnet18(self, input_channels, num_classes):
        """
        Build ResNet18 architecture optimized for CIFAR-10.

        Args:
            input_channels (int): Number of input channels (3 for CIFAR-10)
            num_classes (int): Number of output classes (10 for CIFAR-10)

        Returns:
            nn.ModuleDict: ResNet18 model components
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

        # Initial convolution layer
        conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        inplanes = 64
        layer1, inplanes = make_layer(inplanes, 64, 2)
        layer2, inplanes = make_layer(inplanes, 128, 2, stride=2)
        layer3, inplanes = make_layer(inplanes, 256, 2, stride=2)
        layer4, inplanes = make_layer(inplanes, 512, 2, stride=2)

        # Final layers
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        return nn.ModuleDict({
            'conv1': conv1,
            'bn1': bn1,
            'relu': relu,
            'maxpool': maxpool,
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'layer4': layer4,
            'avgpool': avgpool,
            'fc': fc
        })

    def forward(self, x):
        """
        Forward pass through ResNet18.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.model['conv1'](x)
        x = self.model['bn1'](x)
        x = self.model['relu'](x)
        x = self.model['maxpool'](x)

        x = self.model['layer1'](x)
        x = self.model['layer2'](x)
        x = self.model['layer3'](x)
        x = self.model['layer4'](x)

        x = self.model['avgpool'](x)
        x = torch.flatten(x, 1)
        x = self.model['fc'](x)

        return x

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with weight decay
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self._optimizer = optimizer
        return optimizer
