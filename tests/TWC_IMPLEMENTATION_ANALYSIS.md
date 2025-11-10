# TWC Implementation Comparison Analysis

## Overview
This document provides an in-depth analysis comparing the authors' original TWC implementation (ariel module) with the PyTorch reimplementation (twc/fiuri modules).

## Architecture Comparison

### Authors' Implementation (ariel module)

**Structure:**
- `Model` class: Top-level container
  - `NeuralNetwork`: Contains neurons and connections
  - `Interfaces`: BinaryInterface objects for I/O mapping
- `Neuron` class: Individual neuron with internal/output states
- `Connection` class: Connections between neurons (ChemEx, ChemIn, AGJ/SGJ)

**Key Components:**
1. **NeuralNetwork.doSimulationStep()**: 
   - First computes `computeVnext()` for all neurons (buffers new states)
   - Then commits with `commitComputation()` for all neurons
   - This two-phase update prevents order-dependent results

2. **Neuron Dynamics** (`computeVnext()`):
   - Computes influence from all dendritic connections
   - Clamps stimulus to [-10, 10]
   - Updates buffered states based on threshold comparison
   - Three cases:
     - `S > T`: `E_new = O = S - T`
     - `S == E`: `E_new = E - D`, `O = 0` (decay)
     - Otherwise: `E_new = S`, `O = 0`

3. **Connection Types**:
   - `ChemEx`: Excitatory, adds `w * O_source`
   - `ChemIn`: Inhibitory, subtracts `w * O_source`
   - `AGJ/SGJ`: Gap junction, sign depends on relative states

4. **I/O Interfaces**:
   - `BinaryInterface`: Maps scalar values to positive/negative neuron pairs
   - `IN1`: Position → PLM (pos) / AVM (neg)
   - `IN2`: Velocity → ALM (pos) / PVD (neg)
   - `OUT1`: FWD (pos) / REV (neg) → scalar action

5. **Update Flow** (`Model.Update()`):
   - Runs `simmulationSteps-1` steps (default: 1 step)
   - For each step:
     - Sets interface values from observations
     - Feeds inputs via `feedNN()`
     - Runs `doSimulationStep()`
     - Accumulates output from `getFeedBackNN()`

### User's Implementation (twc/fiuri modules)

**Structure:**
- `TWC` class: PyTorch nn.Module
  - `in_layer`, `hid_layer`, `out_layer`: FIURIModule layers
  - Connection modules: `FiuriDenseConn` (EX/IN), `FiuriSparseGJConn` (GJ)
- `FIURIModule`: Layer of neurons with batched operations
- `FiuriDenseConn`: Dense connections with masks
- `FiuriSparseGJConn`: Sparse gap junction connections

**Key Components:**
1. **Forward Pass** (`TWC.forward()`):
   - Encodes observations to input layer
   - Runs `internal_steps` iterations (configurable)
   - Each iteration processes through layers with connections
   - Returns decoded action

2. **Neuron Dynamics** (`FIURIModule.forward()`):
   - Computes stimulus: `S = clamp(E + chem_influence + gj_influence, -10, 10)`
   - Updates states:
     - `O = ReLU(S - T)` (output)
     - `E = S - T` if `S > T`, else `E - D` if `S == E`, else `S`
   - Batched operations for efficiency

3. **Connection Types**:
   - `FiuriDenseConn`: EX/IN connections with weight masks
   - `FiuriSparseGJConn`: Gap junctions with edge list
   - Weights use `softplus()` to ensure positivity

4. **I/O**:
   - `mcc_obs_encoder`: Direct mapping to 4 input neurons
   - `twc_out_2_mcc_action`: Maps 2 output neurons to action

5. **State Management**:
   - States stored in `_state` dictionary
   - Each layer has `(E, O)` tuple
   - Batched: `(B, N)` tensors

## Key Differences

### 1. Execution Model
- **Ariel**: Single simulation step per `Update()` call (simmulationSteps-1 = 1)
- **Torch**: Configurable `internal_steps` (default: 1, can be increased)

### 2. State Updates
- **Ariel**: Two-phase update (compute, then commit) to avoid order dependence
- **Torch**: Single forward pass with batched operations (order-independent by design)

### 3. Weight Representation
- **Ariel**: Direct weight values (can be negative for IN connections)
- **Torch**: Positive weights via `softplus()`, sign applied in forward pass

### 4. I/O Mapping
- **Ariel**: BinaryInterface maps scalars to pos/neg neuron pairs
- **Torch**: Direct encoding to 4 input neurons (equivalent but different implementation)

### 5. Gap Junction Handling
- **Ariel**: AGJ/SGJ connections with sign based on state comparison
- **Torch**: Sparse GJ connections with sign computed in `_compute_gj_sum()`

### 6. Batching
- **Ariel**: Single sample processing
- **Torch**: Batched processing (B, N) for efficiency

## Neuron Dynamics Equivalence

Both implementations follow the same mathematical model:

1. **Stimulus Computation**:
   - `S = E + Σ(I_j)` where `I_j` depends on connection type
   - Clamped to [-10, 10]

2. **State Updates**:
   - If `S > T`: `E = O = S - T`
   - If `S == E`: `E = E - D`, `O = 0`
   - Otherwise: `E = S`, `O = 0`

3. **Connection Influence**:
   - EX: `+w * O_source`
   - IN: `-w * O_source`
   - GJ: `±w * O_source` (sign based on state comparison)

## Test Strategy

The validation test (`twc_validation.py`) performs:

1. **Weight Synchronization**: Extracts weights from ariel model and sets them in PyTorch model
   - Handles softplus inverse for weight conversion
   - Maps connections by neuron names

2. **Parameter Synchronization**: Syncs thresholds and decay factors

3. **State Comparison**: 
   - Runs both models with same inputs
   - Compares outputs and internal states
   - Tracks differences with tolerance thresholds

4. **Visualization**: 
   - Output scatter plots
   - Error distributions
   - State difference analysis

5. **Detailed Reporting**: 
   - Per-test case differences
   - Summary statistics
   - Match rates

## Potential Sources of Differences

1. **Numerical Precision**: 
   - Ariel uses numpy floats
   - Torch uses torch.float32
   - Small floating-point differences expected

2. **Weight Initialization**:
   - Need to ensure weights are exactly synchronized
   - Softplus inverse calculation may introduce small errors

3. **State Initialization**:
   - Both should start from zero states
   - Need to verify reset behavior matches

4. **Interface Mapping**:
   - BinaryInterface vs direct encoding should be equivalent
   - Need to verify mapping correctness

5. **Simulation Steps**:
   - Must match `internal_steps=1` to ariel's single step

## Validation Criteria

The test considers implementations matching if:
- Output differences < 1e-4 (configurable tolerance)
- State differences < 1e-4 for internal/output states
- Threshold/decay differences < 1e-4

Small differences (< 1e-5) are expected due to numerical precision and implementation details.

