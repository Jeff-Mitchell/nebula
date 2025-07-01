import asyncio
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


class PQLightning(Lightning):
    """
    PQLightning extends Lightning to apply Pruning and Quantization to models.
    
    This class applies model compression techniques before sending models and
    restores them correctly when receiving models to ensure proper functionality.
    """
    
    def __init__(self, model, datamodule, config=None, pruning_ratio=0.3, quantization_bits=8):
        super().__init__(model, datamodule, config)
        
        # Compression parameters
        self.pruning_ratio = pruning_ratio
        self.quantization_bits = quantization_bits
        
        # Store original model state for restoration
        self._original_model_state = None
        self._pruning_masks = {}
        self._quantization_scales = {}
        self._quantization_zero_points = {}
        
        # Track if model is currently compressed
        self._is_compressed = False
        
        logging_training.info(f"PQLightning initialized with pruning_ratio={pruning_ratio}, quantization_bits={quantization_bits}")

    def _store_original_state(self):
        """Store the original model state before compression."""
        if not self._is_compressed:
            self._original_model_state = copy.deepcopy(self.model.state_dict())
            logging_training.debug("Original model state stored")

    def _apply_pruning(self, model_params):
        """
        Apply pruning to model parameters.
        
        Args:
            model_params (OrderedDict): Model parameters to prune
            
        Returns:
            OrderedDict: Pruned model parameters with masks
        """
        pruned_params = copy.deepcopy(model_params)
        self._pruning_masks = {}
        
        for layer_name, param in pruned_params.items():
            if len(param.shape) > 1:  # Only prune weight matrices, not biases
                # Create pruning mask
                mask = torch.ones_like(param, dtype=torch.bool)
                num_elements = param.numel()
                num_prune = int(num_elements * self.pruning_ratio)
                
                if num_prune > 0:
                    # Get indices of smallest magnitude weights
                    flat_param = param.abs().flatten()
                    _, indices = torch.topk(flat_param, k=num_elements - num_prune, largest=True)
                    
                    # Create mask
                    flat_mask = torch.zeros_like(flat_param, dtype=torch.bool)
                    flat_mask[indices] = True
                    mask = flat_mask.view_as(param)
                    
                    # Apply mask
                    pruned_params[layer_name] = param * mask
                    self._pruning_masks[layer_name] = mask
                    
                    logging_training.debug(f"Pruned layer {layer_name}: {num_prune}/{num_elements} elements ({self.pruning_ratio*100:.1f}%)")
        
        return pruned_params

    def _apply_quantization(self, model_params):
        """
        Apply quantization to model parameters.
        
        Args:
            model_params (OrderedDict): Model parameters to quantize
            
        Returns:
            OrderedDict: Quantized model parameters with scale/zero_point info
        """
        quantized_params = copy.deepcopy(model_params)
        self._quantization_scales = {}
        self._quantization_zero_points = {}
        
        for layer_name, param in quantized_params.items():
            if param.numel() > 0:
                # Calculate quantization parameters
                min_val = param.min()
                max_val = param.max()
                
                if max_val > min_val:
                    # Dynamic quantization
                    if self.quantization_bits == 8:
                        scale = (max_val - min_val) / 255.0
                        zero_point = int(round(-min_val / scale))
                        zero_point = max(0, min(255, zero_point))  # Clamp to valid range
                        
                        # Quantize
                        quantized = torch.round(param / scale + zero_point)
                        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
                        
                        # Store quantization info
                        self._quantization_scales[layer_name] = scale
                        self._quantization_zero_points[layer_name] = zero_point
                        quantized_params[layer_name] = quantized
                        
                        logging_training.debug(f"Quantized layer {layer_name}: scale={scale:.6f}, zero_point={zero_point}")
                    else:
                        # For other bit widths, use a simplified approach
                        scale = (max_val - min_val) / ((2 ** self.quantization_bits) - 1)
                        zero_point = int(round(-min_val / scale))
                        
                        quantized = torch.round(param / scale + zero_point)
                        quantized = torch.clamp(quantized, 0, (2 ** self.quantization_bits) - 1).to(torch.int32)
                        
                        self._quantization_scales[layer_name] = scale
                        self._quantization_zero_points[layer_name] = zero_point
                        quantized_params[layer_name] = quantized
        
        return quantized_params

    def _restore_from_pruning(self, model_params):
        """
        Restore model parameters from pruning by applying stored masks.
        
        Args:
            model_params (OrderedDict): Pruned model parameters
            
        Returns:
            OrderedDict: Restored model parameters
        """
        restored_params = copy.deepcopy(model_params)
        
        for layer_name, param in restored_params.items():
            if layer_name in self._pruning_masks:
                mask = self._pruning_masks[layer_name]
                # Apply mask to restore pruned values to zero
                restored_params[layer_name] = param * mask
                logging_training.debug(f"Restored pruning for layer {layer_name}")
        
        return restored_params

    def _restore_from_quantization(self, model_params):
        """
        Restore model parameters from quantization using stored scale/zero_point.
        
        Args:
            model_params (OrderedDict): Quantized model parameters
            
        Returns:
            OrderedDict: Restored model parameters
        """
        restored_params = copy.deepcopy(model_params)
        
        for layer_name, param in restored_params.items():
            if layer_name in self._quantization_scales:
                scale = self._quantization_scales[layer_name]
                zero_point = self._quantization_zero_points[layer_name]
                
                # Dequantize
                if param.dtype == torch.uint8:
                    restored_params[layer_name] = (param.float() - zero_point) * scale
                elif param.dtype == torch.int32:
                    restored_params[layer_name] = (param.float() - zero_point) * scale
                else:
                    # If not quantized, keep as is
                    restored_params[layer_name] = param.float()
                
                logging_training.debug(f"Restored quantization for layer {layer_name}")
        
        return restored_params

    def compress_model_for_transmission(self, model_params):
        """
        Compress model parameters for transmission by applying pruning and quantization.
        
        Args:
            model_params (OrderedDict): Original model parameters
            
        Returns:
            dict: Compressed model data including parameters, masks, and quantization info
        """
        logging_training.info("Compressing model for transmission...")
        
        # Store original state
        self._store_original_state()
        
        # Apply pruning
        pruned_params = self._apply_pruning(model_params)
        
        # Apply quantization
        compressed_params = self._apply_quantization(pruned_params)
        
        # Prepare compression metadata
        compression_data = {
            'parameters': compressed_params,
            'pruning_masks': self._pruning_masks,
            'quantization_scales': self._quantization_scales,
            'quantization_zero_points': self._quantization_zero_points,
            'pruning_ratio': self.pruning_ratio,
            'quantization_bits': self.quantization_bits
        }
        
        self._is_compressed = True
        logging_training.info("Model compression completed")
        
        return compression_data

    def decompress_received_model(self, compression_data):
        """
        Decompress received model parameters by restoring from pruning and quantization.
        
        Args:
            compression_data (dict): Compressed model data
            
        Returns:
            OrderedDict: Restored model parameters
        """
        logging_training.info("Decompressing received model...")
        
        # Extract data
        compressed_params = compression_data['parameters']
        pruning_masks = compression_data.get('pruning_masks', {})
        quantization_scales = compression_data.get('quantization_scales', {})
        quantization_zero_points = compression_data.get('quantization_zero_points', {})
        
        # Store compression info for restoration
        self._pruning_masks = pruning_masks
        self._quantization_scales = quantization_scales
        self._quantization_zero_points = quantization_zero_points
        
        # Restore from quantization
        dequantized_params = self._restore_from_quantization(compressed_params)
        
        # Restore from pruning
        restored_params = self._restore_from_pruning(dequantized_params)
        
        logging_training.info("Model decompression completed")
        
        return restored_params

    def get_model_parameters(self, bytes=False, initialize=False):
        """
        Override to apply compression before returning model parameters.
        """
        if bytes:
            # For byte transmission, compress the model
            original_params = self.model.state_dict()
            compression_data = self.compress_model_for_transmission(original_params)
            return self.serialize_model(compression_data)
        else:
            # For internal use, return original parameters
            return self.model.state_dict()

    def set_model_parameters(self, params, initialize=False):
        """
        Override to handle compressed model parameters.
        """
        try:
            if isinstance(params, dict) and 'parameters' in params:
                # This is compressed data, decompress first
                restored_params = self.decompress_received_model(params)
                self.model.load_state_dict(restored_params)
            else:
                # Regular parameters, load directly
                self.model.load_state_dict(params)
        except Exception as e:
            raise ParameterSettingError("Error setting parameters") from e

    def serialize_model(self, model):
        """
        Override to handle compression data serialization.
        """
        try:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                torch.save(model, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            serialized_data = buffer.getvalue()
            buffer.close()
            del buffer
            return serialized_data
        except Exception as e:
            raise ParameterSerializeError("Error serializing model") from e

    def deserialize_model(self, data):
        """
        Override to handle compression data deserialization.
        """
        try:
            buffer = io.BytesIO(data)
            with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                params_dict = torch.load(f)
            buffer.close()
            del buffer
            
            # Check if this is compressed data
            if isinstance(params_dict, dict) and 'parameters' in params_dict:
                # Decompress the data
                return self.decompress_received_model(params_dict)
            else:
                # Regular parameters
                return OrderedDict(params_dict)
        except Exception as e:
            raise ParameterDeserializeError("Error decoding parameters") from e

    def validate_neighbour_model(self, neighbour_model_param):
        """
        Override to handle compressed neighbor model validation.
        """
        avg_loss = 0
        running_loss = 0
        bootstrap_dataloader = self.datamodule.bootstrap_dataloader()
        num_samples = 0
        neighbour_model = copy.deepcopy(self.model)
        
        # Handle compressed parameters
        if isinstance(neighbour_model_param, dict) and 'parameters' in neighbour_model_param:
            restored_params = self.decompress_received_model(neighbour_model_param)
            neighbour_model.load_state_dict(restored_params)
        else:
            neighbour_model.load_state_dict(neighbour_model_param)

        # enable evaluation mode, prevent memory leaks.
        # no need to switch back to training since model is not further used.
        if torch.cuda.is_available():
            neighbour_model = neighbour_model.to("cuda")
        neighbour_model.eval()

        with torch.no_grad():
            for inputs, labels in bootstrap_dataloader:
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")
                outputs = neighbour_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                running_loss += loss.item()
                num_samples += inputs.size(0)

        avg_loss = running_loss / len(bootstrap_dataloader)
        logging_training.info(f"Computed neighbor loss over {num_samples} data samples")
        return avg_loss

    def get_compression_stats(self):
        """
        Get statistics about the current compression.
        
        Returns:
            dict: Compression statistics
        """
        if not self._is_compressed:
            return {"compressed": False}
        
        total_original_params = 0
        total_compressed_params = 0
        total_pruned_params = 0
        
        for layer_name, param in self.model.state_dict().items():
            total_original_params += param.numel()
            
            if layer_name in self._pruning_masks:
                mask = self._pruning_masks[layer_name]
                total_pruned_params += (mask == 0).sum().item()
        
        compression_ratio = 1.0 - (total_pruned_params / total_original_params) if total_original_params > 0 else 1.0
        
        return {
            "compressed": True,
            "pruning_ratio": self.pruning_ratio,
            "quantization_bits": self.quantization_bits,
            "total_original_params": total_original_params,
            "total_pruned_params": total_pruned_params,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": (1.0 - compression_ratio) * 100
        }

    def cleanup(self):
        """
        Override to clean up compression-related data.
        """
        super().cleanup()
        
        # Clear compression data
        self._original_model_state = None
        self._pruning_masks = {}
        self._quantization_scales = {}
        self._quantization_zero_points = {}
        self._is_compressed = False
        
        logging_training.debug("PQLightning compression data cleaned up")
