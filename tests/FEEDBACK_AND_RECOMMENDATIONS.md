# Feedback on TWC Validation Test Results

## Summary

Your test results show **significant mismatches** between the ariel and torch implementations:
- **Match rate**: 64.55% (312/880 state mismatches)
- **Output differences**: Mean 0.025, Max 0.5
- **Critical pattern**: All hidden/output neurons show exactly **10.0 errors** (clamp boundary)

## Root Cause: Input State Setting Mismatch

### The Problem

**Ariel Implementation:**
```python
# BinaryInterface.feedNN() directly SETS input neuron states
self.positiveNeuron.setInternalState(posPot)  # Overwrites E
self.positiveNeuron.setOutputState(posPot)      # Overwrites O
```

**Torch Implementation:**
```python
# FIURIModule.forward() ADDS encoder values to current state
S = torch.clamp(E + chem_influence, ...)  # E starts at 0, adds encoder values
```

### Why This Matters

1. **Ariel**: Input neurons have their states **directly overwritten** with encoded values
2. **Torch**: Input neurons compute stimulus as `S = E + encoder_values`, then update via normal dynamics
3. **Result**: Input neurons end up with different states, causing cascade of errors

### Evidence from Test Results

- **PLM/AVM errors**: ~1-2 units (input layer mismatch)
- **All hidden neurons**: Exactly 10.0 errors (likely at clamp boundary due to wrong inputs)
- **All output neurons**: Exactly 10.0 errors (propagated from hidden layer)

## Specific Issues Found

### Issue 1: Input Encoding Method

**Ariel's `feedNN()`:**
- Uses hard threshold: `if value >= valleyVal`
- Directly sets both internal AND output states
- One neuron gets encoded value, other gets `minState` (-10)

**Torch's `mcc_obs_encoder()`:**
- Uses sigmoid gates (smooth transition)
- Returns values that are added to state
- Both neurons may receive non-zero values

**Fix Needed**: Make input encoding match exactly, or ensure input layer states are SET (not added to).

### Issue 2: Input Layer Processing

The torch `TWC.forward()` does:
```python
in_out, new_in_state = self.in_layer(ex_in + in_in, state=in_state)
```

This computes `S = E + (ex_in + in_in)`, but ariel sets `E = encoded_value` directly.

**Fix Needed**: For input layer, either:
1. Set states directly before processing, OR
2. Ensure E=0 and verify encoder values match ariel's feedNN output exactly

### Issue 3: The 10.0 Error Pattern

All hidden/output neurons showing exactly 10.0 errors suggests:
- They're at clamp boundary (10.0) in one implementation
- At 0.0 in the other implementation
- This happens because input layer errors cascade through the network

## Recommended Fixes (Priority Order)

### Priority 1: Fix Input State Setting

**Option A** (Recommended): Modify input layer processing to SET states directly

```python
# In TWC.forward(), for input layer:
ex_in, in_in = self.obs_encoder(x, n_inputs=4, device=device)

# Instead of: in_out, new_in_state = self.in_layer(ex_in + in_in, state=in_state)
# Do this:
# Set input neuron states directly (matching ariel feedNN behavior)
# For each input neuron, set E and O directly from encoder values
```

**Option B**: Ensure encoder values match ariel exactly, and verify E=0 before adding

### Priority 2: Verify Encoder Equivalence

Create a test to compare:
- `BinaryInterface.feedNN()` output values
- `mcc_obs_encoder()` output values

They should produce identical values for the same input.

### Priority 3: Add Debug Logging

Add detailed logging to track:
1. Input neuron states after encoding (both implementations)
2. Hidden neuron states after first propagation step
3. Actual weights used (after softplus)
4. Connection influences computed

### Priority 4: Verify Weight Synchronization

The softplus inverse may introduce small errors. Consider:
- Testing with weights=1.0 (use `_set_all_weights_to_one()`)
- Or directly setting weights without softplus for validation

## Immediate Action Items

1. **Create input encoding comparison test**:
   ```python
   # Test that mcc_obs_encoder produces same values as feedNN()
   for obs in test_observations:
       # Get ariel values
       fiu_twc.interfaces['IN1'].setValue(obs[0])
       fiu_twc.interfaces['IN1'].feedNN()
       ariel_plm = fiu_twc.getNeuron('PLM').getInternalState()
       ariel_avm = fiu_twc.getNeuron('AVM').getInternalState()
       
       # Get torch values
       ex_in, in_in = mcc_obs_encoder(torch.tensor([obs]))
       torch_plm = ex_in[0, 1].item()  # PLM is index 1
       torch_avm = in_in[0, 2].item()  # AVM is index 2
       
       # Compare
   ```

2. **Modify input layer to SET states**:
   - Before network propagation, directly set input neuron E and O from encoder values
   - This matches ariel's feedNN() behavior

3. **Re-run validation test**:
   - Should see input layer errors drop to near zero
   - Hidden/output errors should reduce significantly

## Expected Outcomes After Fixes

- **Input layer**: Errors < 1e-5 (near perfect match)
- **Hidden layer**: Errors should drop from 10.0 to < 1.0
- **Output layer**: Errors should drop from 10.0 to < 0.1
- **Output differences**: Should drop from 0.025 mean to < 0.001

## Additional Observations

1. **Test 2 output difference (0.5)**: This is significant and suggests the output decoding may also need verification

2. **Weight synchronization**: The softplus inverse calculation is correct, but small numerical errors may accumulate

3. **Gap junction connections**: Need to verify GJ connections are mapped correctly (PLM->PVC, AVM->AVD)

## Next Steps

1. Implement input state setting fix (Priority 1)
2. Create and run input encoding comparison test
3. Add detailed step-by-step logging
4. Re-run full validation test
5. If issues persist, investigate weight synchronization and GJ connections

The good news is that the neuron dynamics themselves appear correct (based on your single neuron test). The issue is primarily in how inputs are fed into the network and how states are initialized.

