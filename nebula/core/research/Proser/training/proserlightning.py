import logging
import traceback

from nebula.core.training.lightning import Lightning
from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)


class ProserLightning(Lightning):
    """
    Learner with PyTorch Lightning for Proser.

    """

    def __init__(self, model, data, config=None):
        super().__init__(model, data, config)

    def _train_sync(self):
        try:
            self._trainer.fit(self.model, self.datamodule)
            self.model.calibrate(self.datamodule, self._trainer.strategy.root_device)
        except Exception as e:
            logging_training.error(f"Error in _train_sync: {e}")
            tb = traceback.format_exc()
            logging_training.error(f"Traceback: {tb}")
            # If "raise", the exception will be managed by the main thread
