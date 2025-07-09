from nebula.core.optimization.performance.ProserProto.models.mnist.cnn import ProserProtoCNN
from nebula.core.optimization.performance.ProserProto.models.cifar10.resnet import ProserProtoResNet18

# Registrar modelos por dataset
MODELS = {
    'mnist': {
        'ProserProtoCNN': ProserProtoCNN
    },
    'cifar10': {
        'ProserProtoResNet18': ProserProtoResNet18
    }
}

__all__ = ['ProserProtoCNN', 'ProserProtoResNet18', 'MODELS']
