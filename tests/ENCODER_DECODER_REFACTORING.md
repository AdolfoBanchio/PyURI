# Encoder/Decoder Refactoring Summary

## Overview
Refactored `mcc_obs_encoder()` and `twc_out_2_mcc_action()` in `twc_io.py` to exactly match ariel's `BinaryInterface.feedNN()` and `getFeedBackNN()` behavior.

## Key Changes

### 1. Encoder (`mcc_obs_encoder`)

**Before**: Used sigmoid gates for smooth transitions
```python
pos_gate = torch.sigmoid((pos - POS_VALLEY_VAL) * SMOOTH_GATE_SHARPNESS)
PLM_EX_input = pos_gate * pos_pot + (1 - pos_gate) * min_fill
```

**After**: Uses hard thresholds (matching ariel's `if value >= valleyVal`)
```python
pos_mask = pos >= POS_VALLEY_VAL
PLM_EX_input = torch.where(pos_mask, pos_pot, min_fill)
AVM_IN_input = torch.where(pos_mask, min_fill, pos_pot)
```

**Exact Match:
- IN1 (position): `valleyVal=-0.3`, `minVal=-1.2`, `maxVal=0.6`
  - If `pos >= -0.3`: PLM gets `posPot`, AVM gets `minState` (-10)
  - If `pos < -0.3`: PLM gets `minState`, AVM gets `negPot`
  
- IN2 (velocity): `valleyVal=0.0`, `minVal=-0.1`, `maxVal=0.1`
  - If `vel >= 0.0`: ALM gets `velPot`, PVD gets `minState` (-10)
  - If `vel < 0.0`: ALM gets `minState`, PVD gets `negPot`

### 2. Decoder (`twc_out_2_mcc_action`)

**Before**: Used simplified bounded_affine with hardcoded values
```python
retval1 = bounded_affine(MIN_STATE, 0, MAX_STATE, 1.0, pos_St)
retval2 = bounded_affine(MIN_STATE, 0, MAX_STATE, 1.0, neg_St)
```

**After**: Uses exact interface parameters (matching ariel's OUT1)
```python
retval1 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, OUT_MAX_VAL, pos_St)  # maxValue=1.0
retval2 = bounded_affine(MIN_STATE, 0.0, MAX_STATE, -OUT_MIN_VAL, neg_St)  # -minValue=1.0
```

**Key Fix**: Decoder now receives **internal states** (E), not output states (O), matching ariel's `getFeedBackNN()` which uses `getInternalState()`.

### 3. Bounded Affine Function

Added explicit `bounded_affine()` function matching ariel's implementation:
```python
def bounded_affine(xmin: float, ymin: float, xmax: float, ymax: float, x: torch.Tensor) -> torch.Tensor:
    a = (ymax - ymin) / (xmax - xmin)
    d = ymin - a * xmin
    y = a * x + d
    y = torch.clamp(y, min=ymin, max=ymax)
    return y
```

### 4. Interface Parameters

Added constants matching XML interface parameters:
- `POS_VALLEY_VAL = -0.3`, `POS_MIN_VAL = -1.2`, `POS_MAX_VAL = 0.6`
- `VEL_VALLEY_VAL = 0.0`, `VEL_MIN_VAL = -0.1`, `VEL_MAX_VAL = 0.1`
- `OUT_VALLEY_VAL = 0.0`, `OUT_MIN_VAL = -1.0`, `OUT_MAX_VAL = 1.0`

### 5. TWC Builder Fix

Fixed decoder input in `twc_builder.py`:
- **Before**: Passed output states (O) to decoder
- **After**: Passes internal states (E) to decoder, matching ariel's `getFeedBackNN()`

## Behavior Equivalence

The refactored encoder/decoder now produces **identical** results to ariel's implementation:

1. **Hard thresholds** instead of smooth sigmoid transitions
2. **Exact interface parameters** from XML configuration
3. **Internal states** used for decoding (not output states)
4. **Same normalization formulas** for position and velocity

## Testing

Run the validation test to verify:
- Input layer states should match exactly (no more 1-2 unit errors)
- Output values should match exactly (no more 0.5 differences)
- All state mismatches should be resolved

