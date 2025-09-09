import torch

from nebula.core.models.nebulamodel import NebulaModel


class EdgeIIoTsetMLP(NebulaModel):
    def __init__(
        self,
        input_channels=None, # If None, infer with LazyLinear on first forward
        num_classes=15,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__(input_channels if input_channels is not None else -1, num_classes, learning_rate, metrics, confusion_matrix, seed)

        if input_channels is not None:
            self.example_input_array = torch.zeros(1, input_channels)
            self.l1 = torch.nn.Linear(input_channels, 256)
        else:
            # Infer input features at first forward
            self.example_input_array = None
            self.l1 = torch.nn.LazyLinear(256)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        return x

    @classmethod
    def from_partition(cls, partition_handler, learning_rate=1e-3, metrics=None, confusion_matrix=None, seed=None):
        """Factory that builds the model using the dataset partition metadata.

        Expects an `EdgeIIoTsetPartitionHandler`-like object exposing
        `get_feature_dim()` and `get_num_classes()` methods. Falls back to
        basic inference when methods are missing.
        """
        # Infer input feature dimension
        in_dim = None
        if hasattr(partition_handler, "get_feature_dim"):
            in_dim = partition_handler.get_feature_dim()
        if in_dim is None:
            d = getattr(partition_handler, "data", None)
            if d is not None and hasattr(d, "shape") and len(d.shape) >= 2:
                in_dim = int(d.shape[1])
            else:
                try:
                    sample = partition_handler[0][0]
                    in_dim = int(sample.numel()) if hasattr(sample, "numel") else int(len(sample))
                except Exception:
                    in_dim = 48  # sensible default for Edge-IIoTset

        # Infer number of classes
        num_classes = None
        if hasattr(partition_handler, "get_num_classes"):
            num_classes = partition_handler.get_num_classes()
        if num_classes is None:
            try:
                import numpy as np

                targets = getattr(partition_handler, "targets", None)
                num_classes = int(len(np.unique(targets))) if targets is not None else 15
            except Exception:
                num_classes = 15

        return cls(
            input_channels=in_dim,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            seed=seed,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self._optimizer = optimizer
        return optimizer

    def get_learning_rate(self):
        return self.learning_rate

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
