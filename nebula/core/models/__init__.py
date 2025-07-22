from nebula.core.models.mnist.mlp import MNISTModelMLP
from nebula.core.models.mnist.cnn import MNISTModelCNN
from nebula.core.models.fashionmnist.mlp import FashionMNISTModelMLP
from nebula.core.models.fashionmnist.cnn import FashionMNISTModelCNN
from nebula.core.models.emnist.mlp import EMNISTModelMLP
from nebula.core.models.emnist.cnn import EMNISTModelCNN
from nebula.core.models.cifar10.cnn import CIFAR10ModelCNN
from nebula.core.models.cifar10.cnnV2 import CIFAR10ModelCNN_V2
from nebula.core.models.cifar10.cnnV3 import CIFAR10ModelCNN_V3
from nebula.core.models.cifar10.resnet import CIFAR10ModelResNet
from nebula.core.models.cifar10.simplemobilenet import SimpleMobileNetV1
from nebula.core.models.cifar10.fastermobilenet import FasterMobileNet
from nebula.core.models.cifar100.cnn import CIFAR100ModelCNN
from nebula.core.optimization.performance.ProserProto.models.mnist.cnn import ProserProtoCNN
from nebula.core.optimization.performance.ProserProto.models.cifar10.resnet import ProserProtoResNet18
from nebula.core.research.FedProto.models.mnist.FedProtoCNN import FedProtoMNISTModelCNN
from nebula.core.research.FedProto.models.fashionmnist.FedProtoCNN import FedProtoFashionMNISTModelCNN
from nebula.core.research.FedProto.models.cifar10.FedProtoCNN import FedProtoCIFAR10ModelCNN
from nebula.core.research.FedProto.models.cifar10.FedProtoResnet8 import FedProtoCIFAR10ModelResNet8
from nebula.core.research.FedProto.models.cifar100.FedProtoResnet18 import FedProtoCIFAR100ModelResNet18

# Mapping of model names to model classes
MODELS = {
    'MNIST': {
        'MLP': MNISTModelMLP,
        'CNN': MNISTModelCNN,
        'ProserProtoCNN': ProserProtoCNN,
        'FedProtoCNN': FedProtoMNISTModelCNN
    },
    'fashionmnist': {
        'MLP': FashionMNISTModelMLP,
        'CNN': FashionMNISTModelCNN,
        'FedProtoCNN': FedProtoFashionMNISTModelCNN
    },
    'emnist': {
        'MLP': EMNISTModelMLP,
        'CNN': EMNISTModelCNN
    },
    'cifar10': {
        'CNN': CIFAR10ModelCNN,
        'CNNv2': CIFAR10ModelCNN_V2,
        'CNNv3': CIFAR10ModelCNN_V3,
        'ResNet9': CIFAR10ModelResNet,
        'simplemobilenet': SimpleMobileNetV1,
        'fastermobilenet': FasterMobileNet,
        'ProserProtoResNet18': ProserProtoResNet18,
        'FedProtoCNN': FedProtoCIFAR10ModelCNN,
        'FedProtoResNet8': FedProtoCIFAR10ModelResNet8
    },
    'cifar100': {
        'CNN': CIFAR100ModelCNN,
        'FedProtoResNet18': FedProtoCIFAR100ModelResNet18
    }
}

__all__ = [
    'MNISTModelMLP', 'MNISTModelCNN',
    'FashionMNISTModelMLP', 'FashionMNISTModelCNN',
    'EMNISTModelMLP', 'EMNISTModelCNN',
    'CIFAR10ModelCNN', 'CIFAR10ModelCNN_V2', 'CIFAR10ModelCNN_V3',
    'CIFAR10ModelResNet', 'SimpleMobileNetV1', 'FasterMobileNet',
    'CIFAR100ModelCNN',
    'ProserProtoCNN', 'ProserProtoResNet18',
    'FedProtoMNISTModelCNN', 'FedProtoFashionMNISTModelCNN',
    'FedProtoCIFAR10ModelCNN', 'FedProtoCIFAR10ModelResNet8',
    'FedProtoCIFAR100ModelResNet18',
    'MODELS'
]
