import sys
import time
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import torch
from fiuri import build_fiuri_twc, build_fiuri_twc_v2, PyUriTwc_V2
from twc import build_twc, twc_out_2_mcc_action, mcc_obs_encoder
import torch
import time
import os
from pathlib import Path
import torch.utils.benchmark as benchmark
from twc.twc import TWC

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 8192        # Massive batch to saturate GPU
SEQ_LEN = 10             # Typical RL training sequence length
WARMUP_ITERS = 10
PROFILE_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set precision for Ampere+ GPUs (optional but recommended)
torch.set_float32_matmul_precision('high')

def get_model():
    # Initialize model
    model = build_fiuri_twc_v2()
    model.to(DEVICE)
    # Ensure it's in training mode (though dropout is likely 0)
    model.train()
    return model

def run_benchmark(model:PyUriTwc_V2, name, use_compile=False):
    is_twc = isinstance(model, TWC)

    def make_initial_state(requires_grad: bool):
        if is_twc:
            state_dict = model.get_initial_state(BATCH_SIZE, DEVICE)
            if requires_grad:
                state_dict = {
                    k: (E.requires_grad_(), O.requires_grad_())
                    for k, (E, O) in state_dict.items()
                }
            return state_dict
        else:
            init_E, init_O = model.get_initial_state(BATCH_SIZE, DEVICE)
            if requires_grad:
                init_E.requires_grad = True
                init_O.requires_grad = True
            return (init_E, init_O)

    def forward_sequence(obs_seq, state):
        if is_twc:
            actions = []
            cur_state = state
            for t in range(obs_seq.shape[1]):
                obs_t = obs_seq[:, t, :]
                action_t, cur_state = model.forward_bptt(obs_t, cur_state)
                actions.append(action_t)
            return torch.stack(actions, dim=1), cur_state
        else:
            return model.forward_bptt(obs_seq, state)

    print(f"\n{'='*40}")
    print(f"Benchmarking: {name}")
    print(f"Config: Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, Compile={use_compile}")
    print(f"{'='*40}")

    # 1. Compile (if enabled)
    if use_compile:
        print("--> Compiling forward_bptt (mode='reduce-overhead')...")
        t0 = time.time()
        forward_sequence = torch.compile(forward_sequence, mode="default")
        
        # Trigger JIT Compilation with dummy data
        dummy_obs = torch.randn(BATCH_SIZE, SEQ_LEN, 2, device=DEVICE)
        dummy_state = make_initial_state(requires_grad=False)
        with torch.no_grad():
            forward_sequence(dummy_obs, dummy_state)
        torch.cuda.synchronize()
        print(f"--> Compilation finished in {time.time() - t0:.2f}s")

    # 2. Setup Data
    obs_seq = torch.randn(BATCH_SIZE, SEQ_LEN, 2, device=DEVICE)
    init_state = make_initial_state(requires_grad=True)

    # 3. Warmup
    print(f"--> Warming up ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
        actions, _ = forward_sequence(obs_seq, init_state)
        # Simulate a simple backward pass to profile Autograd overhead too
        loss = actions.sum()
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # 4. Profiling Execution
    print(f"--> Running Profiler ({PROFILE_ITERS} iters)...")
    log_dir = Path("out/runs/profiles") / f"{name}_B{BATCH_SIZE}_L{SEQ_LEN}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # We use the PyTorch Profiler to generate the Chrome Trace
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=PROFILE_ITERS),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False # Set to True if you need python stack traces (slower)
    ) as prof:
        
        start_time = time.time()
        for i in range(PROFILE_ITERS):
            # A. Forward BPTT
            actions, _ = forward_sequence(obs_seq, init_state)
            
            # B. Backward (Approximating TD3 critic backward is separate, 
            # this measures Actor gradients specifically)
            loss = actions.mean()
            loss.backward()
            model.zero_grad(set_to_none=True) # efficient zero_grad
            
            prof.step()
            
        torch.cuda.synchronize()
        end_time = time.time()

    # 5. Report Statistics
    total_time = end_time - start_time
    avg_time_ms = (total_time / PROFILE_ITERS) * 1000
    steps_per_sec = (BATCH_SIZE * SEQ_LEN * PROFILE_ITERS) / total_time
    
    print(f"\nRESULT [{name}]:")
    print(f"  Avg Iteration Time: {avg_time_ms:.2f} ms")
    print(f"  Throughput (Agent Steps/sec): {steps_per_sec:.2e}")
    print(f"  Trace saved to: {log_dir}")
    return steps_per_sec

def main():
    print("--- Starting Significant Profiling ---")
    
    # Run Eager Mode
    model_eager = get_model()
    eager_throughput = run_benchmark(model_eager, "Eager_Mode", use_compile=False)
    
    # Run Compiled Mode
    model_compiled = get_model() # Fresh model instance
    compiled_throughput = run_benchmark(model_compiled, "Compiled_Mode", use_compile=True)
    
    model_compiled2 = build_twc(
        obs_encoder=mcc_obs_encoder,
        action_decoder=twc_out_2_mcc_action,
        rnd_init=True,
        use_V2=True,
        log_stats=False
    )
    model_compiled2.to(DEVICE)
    compiled_throughput2 = run_benchmark(model_compiled2, "Compiled_Mode_2", use_compile=True)
    
    # Speedup Summary
    speedup = compiled_throughput / eager_throughput
    speedup2 = compiled_throughput2 / eager_throughput
    print(f"\n{'='*40}")
    print(f"FINAL COMPARISON")
    print(f"{'='*40}")
    print(f"Eager Throughput:    {eager_throughput:.2e} steps/sec")
    print(f"Compiled Throughput: {compiled_throughput:.2e} steps/sec")
    print(f"Compiled Throughput2: {compiled_throughput2:.2e} steps/sec")
    print(f"Speedup Factor:      {speedup:.2f}x")
    print(f"Speedup Factor2:      {speedup2:.2f}x")
    
    if speedup < 1.1:
        print("\nWARNING: Speedup is low. Check if 'reduce-overhead' is supported or batch size is too small.")
    else:
        print("\nSUCCESS: Significant speedup detected. CUDA Graphs are working.")

if __name__ == "__main__":
    main()
