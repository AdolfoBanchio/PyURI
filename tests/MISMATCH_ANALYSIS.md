# TWC Implementation Mismatch Analysis

## Executive Summary

The test results show significant mismatches (64.55% match rate, 312/880 state mismatches). The primary issues are:

1. **Input Encoding Mismatch**: Ariel SETS input neuron states, while Torch ADDS to them
2. **Network Propagation**: Hidden/output neurons showing 10.0 errors (clamp boundary)
3. **Output Differences**: One test case shows 0.5 output difference

## Critical Issues Identified

### Issue 1: Input State Setting vs Adding

**Ariel Implementation:**
```python
# BinaryInterface.feedNN() directly SETS states
self.positiveNeuron.setInternalState(posPot)
self.positiveNeuron.setOutputState(posPot)
self.negativeNeuron.setInternalState(self.minState)
self.negativeNeuron.setOutputState(self.minState)
```

**Torch Implementation:**
```python
# mcc_obs_encoder returns values that are ADDED
in_out, new_in_state = self.in_layer(ex_in + in_in, state=in_state)
# This computes: S = E + (ex_in + in_in), then updates
```

**Problem**: 
- Ariel **overwrites** input neuron states with the encoded values
- Torch **adds** the encoded values to existing states
- This causes input neurons to have different initial states before network propagation

**Impact**: 
- Input layer errors of 1-2 units (PLM, AVM)
- Cascades to all downstream neurons
- All hidden/output neurons show 10.0 errors (likely clamped at boundary or stuck at 0)

### Issue 2: State Initialization After Reset

**Ariel**: After `Reset()`, all neurons are at 0. Then `feedNN()` sets input neurons to encoded values.

**Torch**: After `reset()`, states are None, then initialized to 0. Then encoder values are added.

**Problem**: The input layer should be SET, not added to, to match ariel behavior.

### Issue 3: Network Propagation Structure

Looking at the 10.0 errors in hidden/output neurons:
- These are exactly at the clamp boundary (10.0)
- Suggests neurons are either:
  1. Not being updated at all (stuck at 0 in torch, at 10 in ariel)
  2. Being clamped differently
  3. Receiving different inputs due to input layer mismatch

## Detailed Findings

### Input Layer Mismatches

All test cases show input neuron errors:
- **PLM**: ~1-2 unit errors
- **AVM**: ~1-2 unit errors  
- **ALM/PVD**: Not shown but likely similar

These errors propagate through the network.

### Hidden Layer Mismatches

All hidden neurons show **exactly 10.0 errors**:
- DVA, AVD, PVC, AVA, AVB all show 10.0 internal and output state errors
- This suggests they're at the clamp boundary in one implementation and 0 in the other

### Output Layer Mismatches

Output neurons also show 10.0 errors:
- REV, FWD show 10.0 internal and output state errors
- One test case (Test 2) shows 0.5 output difference

## Root Cause Analysis

### Primary Root Cause: Input State Setting

The fundamental difference is:

1. **Ariel flow**:
   ```
   Reset() → all neurons at 0
   feedNN() → directly sets PLM/AVM/ALM/PVD internal and output states
   doSimulationStep() → propagates from these set states
   ```

2. **Torch flow**:
   ```
   reset() → states initialized to 0
   encoder() → returns values
   in_layer(ex_in + in_in, state) → computes S = E + (ex_in + in_in)
   → If E=0, this works, but the update logic may differ
   ```

The issue is that `FIURIModule.forward()` computes:
```python
S = torch.clamp(E + chem_influence, self.clamp_min, self.clamp_max)
```

Where `chem_influence` includes the encoder values. But ariel sets E and O directly, bypassing the stimulus computation for input neurons.

### Secondary Issues

1. **Weight Synchronization**: May not be perfect due to softplus inverse
2. **Threshold/Decay Sync**: Should be correct but verify
3. **Gap Junction Handling**: Need to verify GJ connections are mapped correctly

## Recommended Fixes

### Fix 1: Match Input State Setting

Modify the Torch implementation to SET input neuron states directly, matching ariel:

**Option A**: Modify `FIURIModule.forward()` to detect input layer and set states directly
**Option B**: Create a special input layer that sets states instead of adding
**Option C**: Modify the encoder to return absolute values and ensure input layer E=0 before adding

**Recommended**: Option C - ensure input layer states are 0 before processing, and verify the encoder values match ariel's feedNN() output exactly.

### Fix 2: Verify Input Encoding Equivalence

The ariel `feedNN()` uses:
```python
if value >= valleyVal:
    corVal = value / maxValue
    posPot = (maxState - minState) * corVal + minState
else:
    corVal = value / (-minValue)
    negPot = (maxState - minState) * (-corVal) + minState
```

The torch `mcc_obs_encoder` uses sigmoid gates. Need to verify these produce identical results.

### Fix 3: Debug Network Propagation

Add detailed logging to compare:
- Input neuron states after encoding
- Hidden neuron states after first propagation step
- Output neuron states
- Connection weights actually used

### Fix 4: Verify Weight Synchronization

The softplus inverse calculation may introduce small errors. Consider:
- Directly setting weights without softplus for validation
- Or verifying the actual weights used match exactly

## Testing Strategy

1. **Isolate Input Layer**: Test input encoding in isolation
2. **Single Step Propagation**: Compare one step of propagation
3. **Weight Verification**: Print actual weights used in both implementations
4. **State Tracing**: Log states at each step to find where divergence occurs

## Next Steps

1. Fix input state setting to match ariel (SET vs ADD)
2. Verify encoder produces identical values to feedNN()
3. Add detailed step-by-step logging
4. Re-run validation test
5. If issues persist, investigate weight synchronization and GJ connections

