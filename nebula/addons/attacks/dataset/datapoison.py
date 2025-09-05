"""
This module contains functions for applying data poisoning techniques,
including the application of noise to tensors and modification of datasets
to simulate poisoning attacks.

Functions:
- apply_noise: Applies noise to a tensor based on the specified noise type and poisoning ratio.
- datapoison: Adds noise to a specified portion of a dataset for data poisoning purposes.
- add_x_to_image: Adds an 'X' mark to the top-left corner of an image.
- poison_to_nlp_rawdata: Poisons NLP data by setting word vectors to zero with a given probability.
"""

import copy
import random

import numpy as np
import torch
from skimage.util import random_noise

from nebula.addons.attacks.dataset.datasetattack import DatasetAttack


class SamplePoisoningAttack(DatasetAttack):
    """
    Implements a data poisoning attack on a training dataset.

    This attack introduces noise or modifies specific data points to influence
    the behavior of a machine learning model.

    Args:
        engine (object): The training engine object, including the associated
                         datamodule.
        attack_params (dict): Attack parameters including:
            - poisoned_percent (float): The percentage of data points to be poisoned.
            - poisoned_ratio (float): The ratio of poisoned data relative to the total dataset.
            - targeted (bool): Whether the attack is targeted at a specific label.
            - target_label (int): The target label for the attack (used if targeted is True).
            - noise_type (str): The type of noise to introduce during the attack.
    """

    def __init__(self, engine, attack_params):
        """
        Initializes the SamplePoisoningAttack with the specified engine and parameters.

        Args:
            engine (object): The training engine object.
            attack_params (dict): Dictionary of attack parameters.
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
        
        if "poisoned_ratio" in attack_params:
            self.poisoned_ratio = float(attack_params["poisoned_ratio"])
        elif "poisoned_noise_percent" in attack_params:
            # Convert percentage to ratio (80.0 -> 0.8)
            self.poisoned_ratio = float(attack_params["poisoned_noise_percent"]) / 100.0
        else:
            raise ValueError("Missing required parameter: either 'poisoned_ratio' or 'poisoned_noise_percent' must be provided")
        
        self.targeted = attack_params.get("targeted", False)
        self.target_label = int(attack_params.get("target_label", 3))
        self.noise_type = attack_params.get("noise_type", "salt")
        
        # Store poisoned_node_percent if provided (for potential future use)
        self.poisoned_node_percent = attack_params.get("poisoned_node_percent")

    def apply_noise(self, t, noise_type, poisoned_ratio):
        """
        Applies noise to a tensor based on the specified noise type and poisoning ratio.

        Args:
            t (torch.Tensor): The input tensor to which noise will be applied.
            noise_type (str): The type of noise to apply. Supported types are:
                - "salt": Salt noise (binary salt-and-pepper noise with only 'salt').
                - "gaussian": Gaussian noise with mean 0 and specified variance.
                - "s&p": Salt-and-pepper noise.
                - "nlp_rawdata": Applies a custom NLP raw data poisoning function.
            poisoned_ratio (float): The ratio or variance of noise to be applied, depending on the noise type.

        Returns:
            torch.Tensor: The tensor with noise applied. If the noise type is not supported,
                          returns the original tensor with an error message printed.

        Raises:
            ValueError: If the specified noise_type is not supported.

        Notes:
           - The "nlp_rawdata" noise type requires the custom `poison_to_nlp_rawdata` function.
           - Noise for types "salt", "gaussian", and "s&p" is generated using `random_noise` from
             the `skimage.util` package, and returned as a `torch.Tensor`.
        """
        if noise_type == "salt":
            return torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        elif noise_type == "gaussian":
            return torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
        elif noise_type == "s&p":
            return torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        elif noise_type == "nlp_rawdata":
            return self.poison_to_nlp_rawdata(t, poisoned_ratio)
        else:
            print("ERROR: poison attack type not supported.")
            return t

    def datapoison(
        self,
        dataset,
        indices,
        poisoned_percent,
        poisoned_ratio,
        targeted=False,
        target_label=3,
        noise_type="salt",
    ):
        """
        Adds noise to a specified portion of a dataset for data poisoning purposes.

        This function applies noise to randomly selected samples within a dataset.
        Noise can be targeted or non-targeted. In non-targeted poisoning, random samples
        are chosen and altered using the specified noise type and ratio. In targeted poisoning,
        only samples with a specified label are altered by adding an 'X' pattern.

        Args:
            dataset (Dataset): The dataset to poison, expected to have `.data` and `.targets` attributes.
            indices (list of int): The list of indices in the dataset to consider for poisoning.
            poisoned_percent (float): The percentage of `indices` to poison, as a fraction (0 <= poisoned_percent <= 1).
            poisoned_ratio (float): The intensity or probability parameter for the noise, depending on the noise type.
            targeted (bool, optional): If True, applies targeted poisoning by adding an 'X' only to samples with `target_label`.
                                       Default is False.
            target_label (int, optional): The label to target when `targeted` is True. Default is 3.
            noise_type (str, optional): The type of noise to apply in non-targeted poisoning. Supported types are:
                                        - "salt": Applies salt noise.
                                        - "gaussian": Applies Gaussian noise.
                                        - "s&p": Applies salt-and-pepper noise.
                                        Default is "salt".

        Returns:
            Dataset: A deep copy of the original dataset with poisoned data in `.data`.

        Raises:
            ValueError: If `poisoned_percent` is not between 0 and 1, or if `noise_type` is unsupported.

        Notes:
            - Non-targeted poisoning randomly selects samples from `indices` based on `poisoned_percent`.
            - Targeted poisoning modifies only samples with `target_label` by adding an 'X' pattern, regardless of `poisoned_ratio`.
        """
        new_dataset = copy.deepcopy(dataset)
        train_data = new_dataset.data
        targets = new_dataset.targets
        num_indices = len(indices)
        if not isinstance(noise_type, str):
            noise_type = noise_type[0]

        if not targeted:
            num_poisoned = int(poisoned_percent * num_indices)
            if num_indices == 0:
                return new_dataset
            if num_poisoned > num_indices:
                return new_dataset
            poisoned_indice = random.sample(indices, num_poisoned)

            for i in poisoned_indice:
                t = train_data[i]
                poisoned = self.apply_noise(t, noise_type, poisoned_ratio)
                train_data[i] = poisoned
        else:
            for i in indices:
                if int(targets[i]) == int(target_label):
                    t = train_data[i]
                    poisoned = self.add_x_to_image(t)
                    train_data[i] = poisoned
        new_dataset.data = train_data
        return new_dataset

    def add_x_to_image(self, img):
        """
        Adds a 10x10 pixel 'X' mark to the top-left corner of an image.

        This function modifies the input image by setting specific pixels in the
        top-left 10x10 region to a high intensity value, forming an 'X' shape.
        Pixels on or below the main diagonal and above the secondary diagonal
        are set to 255 (white).

        Args:
            img (array-like): A 2D array or image tensor representing pixel values.
                              It is expected to be in grayscale, where each pixel
                              has a single intensity value.

        Returns:
            torch.Tensor: A tensor representation of the modified image with the 'X' mark.
        """
        for i in range(0, 10):
            for j in range(0, 10):
                if i + j <= 9 or i == j:
                    img[i][j] = 255
        return torch.tensor(img)

    def poison_to_nlp_rawdata(self, text_data, poisoned_ratio):
        """
        Poisons NLP data by setting word vectors to zero with a given probability.

        This function randomly selects a portion of non-zero word vectors in the
        input text data and sets them to zero vectors based on the specified
        poisoning ratio. This simulates a form of data corruption by partially
        nullifying the information in the input data.

        Args:
            text_data (list of torch.Tensor): A list where each entry is a tensor
                representing a word vector. Non-zero vectors are assumed to represent valid words.
            poisoned_ratio (float): The fraction of non-zero word vectors to set to zero,
                where 0 <= poisoned_ratio <= 1.

        Returns:
            list of torch.Tensor: The modified text data with some word vectors set to zero.

        Raises:
            ValueError: If `poisoned_ratio` is greater than 1 or less than 0.

        Notes:
            - `poisoned_ratio` controls the percentage of non-zero vectors to poison.
            - If `num_poisoned_token` is zero or exceeds the number of non-zero vectors,
              the function returns the original `text_data` without modification.
        """
        non_zero_vector_indice = [i for i in range(0, len(text_data)) if text_data[i][0] != 0]
        non_zero_vector_len = len(non_zero_vector_indice)

        num_poisoned_token = int(poisoned_ratio * non_zero_vector_len)
        if num_poisoned_token == 0:
            return text_data
        if num_poisoned_token > non_zero_vector_len:
            return text_data

        poisoned_token_indice = random.sample(non_zero_vector_indice, num_poisoned_token)
        zero_vector = torch.Tensor(np.zeros(len(text_data[0][0])))
        for i in poisoned_token_indice:
            text_data[i] = zero_vector
        return text_data

    def get_malicious_dataset(self):
        """
        Generates a poisoned dataset based on the specified parameters.

        Returns:
            Dataset: A modified version of the training dataset with poisoned data.
        """
        return self.datapoison(
            self.datamodule.train_set,
            self.datamodule.train_set_indices,
            self.poisoned_percent,
            self.poisoned_ratio,
            self.targeted,
            self.target_label,
            self.noise_type,
        )
