import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchvision import transforms
import torch
import numpy as np

from nebula.core.datasets.nebuladataset import NebulaDataset, NebulaPartitionHandler


class EdgeIIoTsetPartitionHandler(NebulaPartitionHandler):
    def __init__(self, file_path, prefix, config, empty=False):
        super().__init__(file_path, prefix, config, empty)

        # Custom transform for Edge-IIoTset
        self.transform = None # Will be defined later
        self.target_transform = None

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def get_feature_dim(self) -> int | None:
        """Infer and return the feature dimensionality for Edge-IIoTset.

        Tries in this order:
        - If `self.data` has shape and is 2D, return second dimension.
        - Otherwise, fetch one sample and infer from its flattened size or length.
        Returns None if it cannot be inferred.
        """
        d = getattr(self, "data", None)
        try:
            if d is not None and hasattr(d, "shape") and len(d.shape) >= 2:
                return int(d.shape[1])
        except Exception:
            pass

        try:
            sample = self[0][0]
            if hasattr(sample, "numel"):
                return int(sample.numel())
            return int(len(sample))
        except Exception:
            return None

    def get_num_classes(self) -> int | None:
        """Return number of classes stored in the partition if available.

        Falls back to computing it from targets when attribute is missing.
        """
        try:
            if getattr(self, "num_classes", None):
                return int(self.num_classes)
        except Exception:
            pass

        try:
            import numpy as np

            t = getattr(self, "targets", None)
            if t is None:
                return None
            return int(len(np.unique(t)))
        except Exception:
            return None


class EdgeIIoTsetDataset(NebulaDataset):
    def __init__(
        self,
        num_classes=15, # Will be adjusted after data inspection
        partitions_number=1,
        batch_size=32,
        num_workers=4,
        iid=True,
        partition="dirichlet",
        partition_parameter=0.5,
        seed=42,
        config_dir=None,
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
            # Use array storage for Edge-IIoT to keep numeric features compact
            partition_storage_mode="arrays",
        )
        self.train_set = None
        self.test_set = None

    def initialize_dataset(self):
        if self.train_set is None:
            self.download_and_load_dataset()

        self.data_partitioning(plot=True)

    def download_and_load_dataset(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)

        # Download from Kaggle
        dataset_path = os.path.join(data_dir, "Edge-IIoTset-dataset.csv")
        if not os.path.exists(dataset_path):
            logging.info("Downloading Edge-IIoTset dataset...")
            # zip_path = os.path.join(data_dir, "edgeiiot.zip")
            # # Download from a public URL to avoid Kaggle API dependency
            # os.system(f'curl -L -o {zip_path} "https://www.kaggle.com/api/v1/datasets/download/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"')
            # os.system(f"unzip {zip_path} -d {data_dir}")
            # os.system(f"rm {zip_path}")

            # The downloaded file might have a different name, so we find it and rename it
            for file in os.listdir(data_dir):
                if file.endswith(".csv") and file != "Edge-IIoTset-dataset.csv":
                    os.rename(os.path.join(data_dir, file), dataset_path)
                    break

        # Load and preprocess data
        df = pd.read_csv(dataset_path, encoding='utf-8', low_memory=False)

        # Preprocessing steps from notebook
        df.drop(['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
                'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp',
                'http.request.uri.query', 'tcp.options', 'tcp.payload', 'tcp.srcport',
                'tcp.dstport', 'udp.port', 'mqtt.msg'], axis=1, inplace=True)

        df.dropna(inplace=True)

        # Correct dtypes
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'Attack_type':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        categorical_cols = ['Attack_type']

        # Encoding categorical features
        label_encoders = {}
        for column in categorical_cols:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        attack_type_classes = label_encoders['Attack_type'].classes_

        features = df.drop('Attack_type', axis=1)
        labels = df['Attack_type']

        # Verify feature count
        # assert features.shape[1] == 48, f"Expected 48 features, but got {features.shape[1]}"

        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=self.seed, stratify=labels)

        # Scaling numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets, classes):
                self.data = data
                self.targets = targets
                self.classes = classes

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        self.train_set = TorchDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long), attack_type_classes)
        self.test_set = TorchDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long), attack_type_classes)
        self.num_classes = len(np.unique(y_train))


    def generate_non_iid_map(self, dataset, partition="dirichlet", partition_parameter=0.5, num_clients=None):
        if partition == "dirichlet":
            partitions_map = self.dirichlet_partition(dataset, alpha=partition_parameter, n_clients=num_clients)
        elif partition == "percent":
            partitions_map = self.percentage_partition(dataset, percentage=partition_parameter, n_clients=num_clients)
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
