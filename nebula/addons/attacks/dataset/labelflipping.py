"""
This module provides a function for label flipping in datasets, allowing for the simulation of label noise
as a form of data poisoning. The main function modifies the labels of specific samples in a dataset based
on a specified percentage and target conditions.

Function:
- labelFlipping: Flips the labels of a specified portion of a dataset to random values or to a specific target label.
"""

import copy
import random

import torch

from nebula.addons.attacks.dataset.datasetattack import DatasetAttack


class LabelFlippingAttack(DatasetAttack):
    """
    Implements an attack that flips the labels of a portion of the training dataset.

    This attack alters the labels of certain data points in the training set to
    mislead the training process.
    """

    def __init__(self, engine, attack_params):
        """
        Initializes the LabelFlippingAttack with the engine and attack parameters.

        Args:
            engine: The engine managing the attack context.
            attack_params (dict): Parameters for the attack, including the percentage of
                                  poisoned data, targeting options, and label specifications.
        """
        try:
            round_start = int(attack_params["round_start_attack"])
            round_stop = int(attack_params["round_stop_attack"])
            attack_interval = int(attack_params["attack_interval"])
        except KeyError as e:
            raise ValueError(f"Missing required attack parameter: {e}")
        except ValueError:
            raise ValueError("Invalid value in attack_params. Ensure all values are integers.")

        super().__init__(engine, round_start, round_stop, attack_interval)
        self.datamodule = engine._trainer.datamodule
        
        # Handle both old and new parameter names for backward compatibility
        if "poisoned_percent" in attack_params:
            self.poisoned_percent = float(attack_params["poisoned_percent"])
        elif "poisoned_sample_percent" in attack_params:
            # Convert percentage to ratio (80.0 -> 0.8)
            self.poisoned_percent = float(attack_params["poisoned_sample_percent"]) / 100.0
        else:
            raise ValueError("Missing required parameter: either 'poisoned_percent' or 'poisoned_sample_percent' must be provided")
        
        self.targeted = attack_params.get("targeted", False)
        self.target_label = int(attack_params.get("target_label", 4))
        self.target_changed_label = int(attack_params.get("target_changed_label", 7))
        
        # Store poisoned_node_percent if provided (for potential future use)
        self.poisoned_node_percent = attack_params.get("poisoned_node_percent")

    def labelFlipping(
        self,
        dataset,
        indices,
        poisoned_percent=0,
        targeted=False,
        target_label=4,
        target_changed_label=7,
    ):
        """
        Flips the labels of a specified portion of a dataset to random values or to a specific target label.

        This function modifies the labels of selected samples in the dataset based on the specified
        poisoning percentage. Labels can be flipped either randomly or targeted to change from a specific
        label to another specified label.

        Args:
            dataset (Dataset): The dataset containing training data, expected to be a PyTorch dataset
                               with a `.targets` attribute.
            indices (list of int): The list of indices in the dataset to consider for label flipping.
            poisoned_percent (float, optional): The ratio of labels to change, expressed as a fraction
                                                (0 <= poisoned_percent <= 1). Default is 0.
            targeted (bool, optional): If True, flips only labels matching `target_label` to `target_changed_label`.
                                       Default is False.
            target_label (int, optional): The label to change when `targeted` is True. Default is 4.
            target_changed_label (int, optional): The label to which `target_label` will be changed. Default is 7.

        Returns:
            Dataset: A deep copy of the original dataset with modified labels in `.targets`.

        Raises:
            ValueError: If `poisoned_percent` is not between 0 and 1, or if `flipping_percent` is invalid.

        Notes:
            - When not in targeted mode, labels are flipped for a random selection of indices based on the specified
              `poisoned_percent`. The new label is chosen randomly from the existing classes.
            - In targeted mode, labels that match `target_label` are directly changed to `target_changed_label`.
        """
        new_dataset = copy.deepcopy(dataset)

        targets = torch.tensor(new_dataset.targets) if isinstance(new_dataset.targets, list) else new_dataset.targets

        num_indices = len(indices)
        class_list = list(set(targets.tolist()))
        if not targeted:
            num_flipped = int(poisoned_percent * num_indices)
            if num_indices == 0:
                return new_dataset
            if num_flipped > num_indices:
                return new_dataset
            flipped_indice = random.sample(indices, num_flipped)

            for i in flipped_indice:
                t = targets[i]
                flipped = torch.tensor(random.sample(class_list, 1)[0])
                while t == flipped:
                    flipped = torch.tensor(random.sample(class_list, 1)[0])
                targets[i] = flipped
        else:
            for i in indices:
                if int(targets[i]) == int(target_label):
                    targets[i] = torch.tensor(target_changed_label)
        new_dataset.targets = targets
        return new_dataset

    def get_malicious_dataset(self):
        """
        Creates a malicious dataset by flipping the labels of selected data points.

        Returns:
            Dataset: The modified dataset with flipped labels.
        """
        return self.labelFlipping(
            self.datamodule.train_set,
            self.datamodule.train_set_indices,
            self.poisoned_percent,
            self.targeted,
            self.target_label,
            self.target_changed_label,
        )
