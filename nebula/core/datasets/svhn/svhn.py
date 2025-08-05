import os

from PIL import Image
from torchvision import transforms
from torchvision.datasets import SVHN

from nebula.core.datasets.nebuladataset import NebulaDataset, NebulaPartitionHandler


class SVHNWrapper:
    """
    Wrapper for SVHN dataset to normalize the interface.
    SVHN uses 'labels' attribute while other datasets use 'targets'.
    """
    def __init__(self, svhn_dataset):
        self._dataset = svhn_dataset
        # Create targets attribute pointing to labels for compatibility
        self.targets = svhn_dataset.labels
        # Create classes attribute for plotting (SVHN has digits 0-9)
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped dataset
        return getattr(self._dataset, name)

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


class SVHNPartitionHandler(NebulaPartitionHandler):
    def __init__(self, file_path, prefix, config, empty=False):
        super().__init__(file_path, prefix, config, empty)

        # Custom transform for SVHN (similar to CIFAR10 but specific normalization)
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True),
        ])

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)

        # SVHN from torchvision returns a tuple (image, target)
        if isinstance(data, tuple):
            img, _ = data
        else:
            img = data

        # Only convert if not already a PIL image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHNDataset(NebulaDataset):
    def __init__(
        self,
        num_classes=10,
        partitions_number=1,
        batch_size=32,
        num_workers=4,
        iid=True,
        partition="dirichlet",
        partition_parameter=0.5,
        seed=42,
        config_dir=None,
        remove_classes_count=0,
    ):
        super().__init__(
            num_classes=num_classes,
            partitions_number=partitions_number,
            batch_size=batch_size,
            num_workers=num_workers,
            iid=iid,
            partition=partition,
            partition_parameter=partition_parameter,
            seed=seed,
            config_dir=config_dir,
            remove_classes_count=remove_classes_count,
        )

    def initialize_dataset(self):
        # Load SVHN train dataset
        if self.train_set is None:
            self.train_set = self.load_svhn_dataset(split="train")
        if self.test_set is None:
            self.test_set = self.load_svhn_dataset(split="test")

        self.data_partitioning(plot=True)

    def load_svhn_dataset(self, split="train"):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        svhn_dataset = SVHN(
            data_dir,
            split=split,
            download=True,
        )
        # Wrap the dataset to normalize the interface (labels -> targets)
        return SVHNWrapper(svhn_dataset)

    def generate_non_iid_map(self, dataset, partition="dirichlet", partition_parameter=0.5, num_clients=None):
        if partition == "dirichlet":
            partitions_map = self.dirichlet_partition(dataset, alpha=partition_parameter, n_clients=num_clients)
        elif partition == "percent":
            partitions_map = self.percentage_partition(dataset, percentage=float(partition_parameter), n_clients=num_clients)
        else:
            raise ValueError(f"Partition {partition} is not supported for Non-IID map")

        return partitions_map

    def generate_iid_map(self, dataset, partition="balancediid", partition_parameter=2, num_clients=None):
        if partition == "balancediid":
            partitions_map = self.balanced_iid_partition(dataset, n_clients=num_clients)
        elif partition == "unbalancediid":
            partitions_map = self.unbalanced_iid_partition(
                dataset, imbalance_factor=partition_parameter, n_clients=num_clients
            )
        else:
            raise ValueError(f"Partition {partition} is not supported for IID map")

        return partitions_map
