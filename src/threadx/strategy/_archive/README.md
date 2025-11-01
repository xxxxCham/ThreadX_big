# ThreadX Strategy - Educational Examples Archive

This directory contains archived educational and example code for strategy implementations.

## Contents

### `gpu_examples.py` (588 lines)
**GPU-Accelerated Strategy Implementation Example**

- **Purpose**: Demonstrates GPU acceleration patterns for trading strategies
- **Contains**: `GPUAcceleratedBBAtr` class extending `BBAtrStrategy`
- **Features**:
  - Multi-GPU distribution pattern
  - GPU-accelerated indicator calculations
  - Educational reference for GPU integration
- **Status**: Archived - Not used in active codebase
- **Archive Date**: 2025-10-31
- **Reason**: Educational material to reduce production code clutter while preserving reference

## Usage

Reference this code if you need to:
- Understand GPU acceleration patterns in ThreadX
- Implement GPU-based strategy variants
- See multi-GPU distribution examples

Note: This is example/educational code. For production GPU strategies, refer to the main ThreadX GPU support in the core modules.

## Integration

To restore to active codebase:
```bash
mv gpu_examples.py ../gpu_examples.py
```

Then update imports if needed in active code.

---

**Archive maintained for reference**: See [PHASE_2_INVESTIGATION_REPORT.md](../../PHASE_2_INVESTIGATION_REPORT.md) for cleanup details.
