import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

from nebula.core.datasets.nebuladataset import NebulaDataset, NebulaPartitionHandler


class EdgeIIoTsetBinaryPartitionHandler(NebulaPartitionHandler):
    def __init__(self, file_path, prefix, config, empty=False):
        super().__init__(file_path, prefix, config, empty)
        self.transform = None
        self.target_transform = None

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def get_feature_dim(self) -> int | None:
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
        try:
            if getattr(self, "num_classes", None):
                return int(self.num_classes)
        except Exception:
            pass
        try:
            t = getattr(self, "targets", None)
            if t is None:
                return None
            return int(len(np.unique(t)))
        except Exception:
            return None


class EdgeIIoTsetBinaryDataset(NebulaDataset):
    def __init__(
        self,
        num_classes=2,
        partitions_number=1,
        batch_size=32,
        num_workers=4,
        iid=True,
        partition="dirichlet",
        partition_parameter=0.5,
        seed=42,
        config_dir=None,
        undersample_normal=True,
        balance_partitions=True,
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
            partition_storage_mode="arrays",
        )
        self.train_set = None
        self.test_set = None
        self.undersample_normal = undersample_normal
        self.balance_partitions = balance_partitions

    def initialize_dataset(self):
        if self.train_set is None:
            self.download_and_load_dataset()
        self._data_partitioning_edge(plot=True)

    def _data_partitioning_edge(self, plot=False):
        import logging as _logging
        _logging.info(
            f"Partitioning data for {self.__class__.__name__} | Partitions: {self.partitions_number} | IID: {self.iid} | Partition: {self.partition} | Partition parameter: {self.partition_parameter}"
        )

        if self.iid:
            self.train_indices_map = self.generate_iid_map(self.train_set)
        else:
            self.train_indices_map = self.generate_non_iid_map(
                self.train_set, partition=self.partition, partition_parameter=self.partition_parameter
            )

        self.test_indices_map = self.get_test_indices_map()
        self.local_test_indices_map = self.get_local_test_indices_map()

        if self.balance_partitions:
            try:
                self._balance_normal_per_participant()
            except Exception as e:
                _logging.warning(f"[EdgeIIoTsetBinaryDataset] Partition balancing skipped due to error: {e}")

        if plot:
            self.plot_data_distribution("train", self.train_set, self.train_indices_map)
            self.plot_all_data_distribution("train", self.train_set, self.train_indices_map)
            self.plot_data_distribution("local_test", self.test_set, self.local_test_indices_map)
            self.plot_all_data_distribution("local_test", self.test_set, self.local_test_indices_map)

        self.save_partitions()

    def _balance_normal_per_participant(self):
        import numpy as np
        rng = np.random.RandomState(self.seed)

        # In this binary variant we define classes as ["Normal", "Attack"], where normal_idx=0
        normal_idx = 0
        for participant, indices in list(self.train_indices_map.items()):
            if not indices:
                continue
            idx_arr = np.array(indices, dtype=int)
            train_targets_arr = getattr(self.train_set, 'targets', None)
            if train_targets_arr is not None:
                try:
                    targets_arr = np.array(train_targets_arr)[idx_arr]
                except Exception:
                    targets_arr = np.array([self.train_set[i][1] for i in idx_arr.tolist()])
            else:
                targets_arr = np.array([self.train_set[i][1] for i in idx_arr.tolist()])

            normal_mask = (targets_arr == normal_idx)
            normal_count = int(normal_mask.sum())
            attacks_count = int(len(targets_arr) - normal_count)
            if normal_count == 0 or attacks_count == 0:
                continue

            target_normal = min(normal_count, attacks_count)
            if target_normal == normal_count:
                continue

            normal_idx_positions = np.where(normal_mask)[0]
            keep_normal_positions = rng.choice(normal_idx_positions, size=target_normal, replace=False)
            attack_positions = np.where(~normal_mask)[0]
            new_positions = np.concatenate([attack_positions, keep_normal_positions])
            rng.shuffle(new_positions)
            new_indices = idx_arr[new_positions].tolist()
            self.train_indices_map[participant] = new_indices

            logging.info(
                f"[EdgeIIoTsetBinaryDataset] Participant {participant}: normal {normal_count} -> {target_normal}, attacks={attacks_count}, total={len(idx_arr)} -> {len(new_indices)}"
            )

    def download_and_load_dataset(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)

        dataset_path = os.path.join(data_dir, "Edge-IIoTset-dataset.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Edge-IIoTset dataset not found at {dataset_path}. "
                "Use the Deployment page 'Download dataset' button or call the controller endpoint "
                "POST /datasets/edge-iiotset/download to prepare it."
            )

        df = pd.read_csv(dataset_path, encoding='utf-8', low_memory=False)

        # Drop columns as in multiclass variant
        df.drop([
            'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
            'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp',
            'http.request.uri.query', 'tcp.options', 'tcp.payload', 'tcp.srcport',
            'tcp.dstport', 'udp.port', 'mqtt.msg'
        ], axis=1, inplace=True)

        df.dropna(inplace=True)

        # Coerce non-label object columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'Attack_type':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        # Build binary labels: Normal/Benign -> 0, Attack -> 1
        labels_raw = df['Attack_type'].astype(str)
        normal_mask = labels_raw.str.lower().str.contains('normal') | labels_raw.str.lower().str.contains('benign')
        y = (~normal_mask).astype(int)  # 0=Normal, 1=Attack

        features = df.drop('Attack_type', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=self.seed, stratify=y
        )

        if self.undersample_normal:
            try:
                X_train, y_train = self._undersample_normal_class_binary(X_train, y_train)
            except Exception as e:
                logging.warning(f"[EdgeIIoTsetBinaryDataset] Undersampling skipped due to error: {e}")

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

        classes = np.array(["Normal", "Attack"])  # index 0 -> Normal, 1 -> Attack
        self.train_set = TorchDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long), classes)
        self.test_set = TorchDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long), classes)
        self.num_classes = 2

    def _undersample_normal_class_binary(self, X_train, y_train):
        """Undersample class 0 (Normal) to roughly match class 1 (Attack)."""
        value_counts = y_train.value_counts()
        normal_count = int(value_counts.get(0, 0))
        attack_count = int(value_counts.get(1, 0))
        if normal_count == 0 or attack_count == 0:
            return X_train, y_train

        target_normal = min(normal_count, attack_count)
        normal_mask = (y_train == 0)
        Xn = X_train[normal_mask]
        yn = y_train[normal_mask]
        Xa = X_train[~normal_mask]
        ya = y_train[~normal_mask]

        sampled_normal = yn.sample(n=target_normal, random_state=self.seed).index
        Xn_sampled = Xn.loc[sampled_normal]
        yn_sampled = yn.loc[sampled_normal]

        X_bal = pd.concat([Xa, Xn_sampled], axis=0)
        y_bal = pd.concat([ya, yn_sampled], axis=0)

        shuffled_idx = y_bal.sample(frac=1.0, random_state=self.seed).index
        X_bal = X_bal.loc[shuffled_idx]
        y_bal = y_bal.loc[shuffled_idx]

        logging.info(
            f"[EdgeIIoTsetBinaryDataset] Undersampled normal class from {normal_count} to {target_normal} (attacks={attack_count}, total_train={len(y_train)} -> {len(y_bal)})"
        )
        return X_bal, y_bal

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
            partitions_map = self.unbalanced_iid_partition(dataset, imbalance_factor=partition_parameter, n_clients=num_clients)
        else:
            raise ValueError(f"Partition {partition} is not supported for IID map")
        return partitions_map
