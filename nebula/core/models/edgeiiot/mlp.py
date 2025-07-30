import torch

from nebula.core.models.nebulamodel import NebulaModel


class EdgeIIoTsetMLP(NebulaModel):
    def __init__(
        self,
        input_channels=48, # Number of features in Edge-IIoTset
        num_classes=15,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.example_input_array = torch.zeros(1, input_channels)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.l1 = torch.nn.Linear(input_channels, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self._optimizer = optimizer
        return optimizer

    def get_learning_rate(self):
        return self.learning_rate

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
