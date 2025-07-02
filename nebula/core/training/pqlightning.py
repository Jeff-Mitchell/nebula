import copy
import gc
import gzip
import hashlib
import io
import logging
import os
import pickle
import traceback
from collections import OrderedDict

import torch
import torch.nn.utils.prune as prune
import torch.quantization as quantization
from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary, ProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch.nn import functional as F

from nebula.config.config import TRAINING_LOGGER
from nebula.core.utils.deterministic import enable_deterministic
from nebula.core.utils.nebulalogger_tensorboard import NebulaTensorBoardLogger
from nebula.core.nebulaevents import TestMetricsEvent
from nebula.core.eventmanager import EventManager
from nebula.core.training.lightning import Lightning, NebulaProgressBar, ParameterSerializeError, ParameterDeserializeError, ParameterSettingError

logging_training = logging.getLogger(TRAINING_LOGGER)

class ParameterQuantizationSettingError(Exception):
    """Custom exception for errors setting model parameters."""


class SerializationError(Exception):
    """Custom exception for errors serializing model parameters."""


class PQLightning(Lightning):
    """
    Learner with PyTorch Lightning. Implements quantization and pruningof model parameters.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None):
        super().__init__(model, data, config)

    def serialize_model(self, model):

        # From https://pytorch.org/docs/stable/notes/serialization.html
        try:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                torch.save(model, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            return buffer.getvalue()
        except Exception as e:
            raise SerializationError("[PQLightning Learner] Error serializing model") from e

    def set_model_parameters(self, params, initialize=False):
        if initialize:
            self.model.load_state_dict(params)
            return

        # Convert parameters back to float32
        logging_training.info("[PQLightning Learner] Decoding parameters...")

        try:
            self.model.load_state_dict(params)
        except Exception as e:
            raise ParameterQuantizationSettingError("[PQLightning Learner] Error setting parameters") from e

    def get_model_parameters(self, bytes=False, initialize=False):
        if initialize:
            if bytes:
                return self.serialize_model(self.model.state_dict())
            return self.model.state_dict()

        model = self.model.state_dict()
        # Convert parameters to float16 before saving to reduce data size
        model = {k: v.half() for k, v in model.items()}

        if bytes:
            return self.serialize_model(model)
        return model