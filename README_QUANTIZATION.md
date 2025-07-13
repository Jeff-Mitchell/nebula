# NEBULA Quantization Feature

## Overview

This feature adds configurable quantization support to NEBULA's federated learning system. Users can now control whether model parameters are quantized (converted to float16) during transmission, which can significantly reduce bandwidth usage and memory consumption.

## Quick Start

### Enable Quantization

Add `"use_quantization": true` to your configuration:

```json
{
  "training_args": {
    "trainer": "lightning",
    "epochs": 1,
    "use_quantization": true
  }
}
```

### Disable Quantization

Add `"use_quantization": false` or omit the field entirely:

```json
{
  "training_args": {
    "trainer": "lightning",
    "epochs": 1,
    "use_quantization": false
  }
}
```

## Technical Details

### Implementation

The quantization feature works by:

1. **Reading Configuration**: The system reads the `use_quantization` field from `training_args`
2. **Trainer Selection**: Based on the value, it selects either:
   - `PQLightning` (with quantization) - converts parameters to float16
   - `Lightning` (without quantization) - keeps parameters in float32
3. **Parameter Processing**: When quantization is enabled, model parameters are converted to float16 before transmission

### Code Changes

Modified files:
- `nebula/core/node.py` - Main entry point logic
- `nebula/node.py` - Alternative entry point logic

### Key Methods

The quantization is implemented in `PQLightning.get_model_parameters()`:

```python
def get_model_parameters(self, bytes=False, initialize=False):
    if initialize:
        return self.model.state_dict()
    
    model = self.model.state_dict()
    # Convert parameters to float16 before saving to reduce data size
    model = {k: v.half() for k, v in model.items()}
    return model
```

## Performance Impact

### With Quantization Enabled
- **Bandwidth**: ~50% reduction in transmission size
- **Memory**: ~50% reduction in memory usage
- **Precision**: Reduced numerical precision (float16 vs float32)

### With Quantization Disabled
- **Bandwidth**: Standard transmission size
- **Memory**: Standard memory usage
- **Precision**: Full numerical precision (float32)

## Use Cases

### Enable Quantization When:
- Network bandwidth is limited
- Memory constraints are present
- Slight precision loss is acceptable
- Working with large models

### Disable Quantization When:
- Maximum precision is required
- Network bandwidth is not a concern
- Working with small models
- Debugging numerical issues

## Compatibility

- **Backward Compatible**: Existing configurations work without changes
- **Default Behavior**: If `use_quantization` is not specified, quantization is disabled
- **Trainer Support**: Only affects `lightning` trainer type

## Examples

See `example_config_quantization.json` for a complete configuration example.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure both `Lightning` and `PQLightning` are available
2. **Configuration Errors**: Verify the `use_quantization` field is in `training_args`
3. **Precision Issues**: If you encounter numerical instability, try disabling quantization

### Debugging

To verify quantization is working:

1. Check the trainer type in logs: `PQLightning` vs `Lightning`
2. Monitor network traffic size
3. Check parameter data types in model state dict

## Contributing

When modifying this feature:

1. Maintain backward compatibility
2. Update both entry points (`nebula/core/node.py` and `nebula/node.py`)
3. Test with both quantization enabled and disabled
4. Update documentation

## References

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [Float16 vs Float32 Precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) 