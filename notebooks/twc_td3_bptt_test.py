# %%
import sys
from pathlib import Path
SRC_ROOT = "../src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# %%
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import gymnasium as gym
from collections import deque
import twc.twc_io
import json
import os
from typing import Callable
from torch import nn
from fiuri import FIURIModule, FiuriDenseConn, FiuriSparseGJConn
from twc.w_builder import build_tw_matrices


# %%
class TWC_BPTT(nn.Module):
    """
    Refactored TWC Module for explicit BPTT.
    
    - forward(): Used for BPTT. Accepts an observation 'x' and an explicit 'state_in',
                 returns (action, state_out). This is a "pure" function.
                 
    - get_action_from_obs(): Used for evaluation/sampling. Manages 'self._state'
                             internally. This is stateful.
                             
    - reset(): Resets the internal 'self._state' for evaluation.
    - get_initial_state(): Gets a batched, zeroed state for starting BPTT.
    """
    def __init__(self, in_layer, hid_layer, out_layer, in2hid_IN, in2hid_GJ,
                 hid_IN, hid_EX, hid2out_EX, obs_encoder, action_decoder,
                 internal_steps=1, log_stats=False):
        super().__init__()
        # neuron layers
        self.in_layer = in_layer
        self.hid_layer = hid_layer
        self.out_layer = out_layer

        # connections
        self.in2hid_IN = in2hid_IN
        self.in2hid_GJ = in2hid_GJ
        self.hid_IN = hid_IN
        self.hid_EX = hid_EX
        self.hid2out = hid2out_EX

        # I/O
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder

        # MONITOR
        self.log = log_stats
        self.monitor = {"in": [], "hid": [], "out": []}
        self.internal_steps = internal_steps
        
        # Internal state for get_action_from_obs (evaluation)
        self._state = None

    def _init_layer_state(self, layer: FIURIModule, batch_size: int, device, dtype):
        E0 = torch.full((batch_size, layer.num_cells), layer._init_E, device=device, dtype=dtype)
        O0 = torch.full((batch_size, layer.num_cells), layer._init_O, device=device, dtype=dtype)
        return (E0, O0)

    def get_initial_state(self, batch_size: int, device, dtype=torch.float32) -> dict:
        """Returns a batched, zeroed state dict for starting a BPTT unroll."""
        return {
            "in": self._init_layer_state(self.in_layer, batch_size, device, dtype),
            "hid": self._init_layer_state(self.hid_layer, batch_size, device, dtype),
            "out": self._init_layer_state(self.out_layer, batch_size, device, dtype),
        }

    def forward(self, x: torch.Tensor, state_in: dict) -> torch.Tensor | dict:
        """
        Performs one recurrent step *for training*. (Pure function)
        
        Args:
            x (torch.Tensor): The observation, e.g., (B, obs_dim).
            state_in (dict): The explicit state from the previous time step.
            
        Returns:
            (torch.Tensor, dict): A tuple of (action, state_out).
        """
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
            
        ex_in, in_in = self.obs_encoder(x, n_inputs=4, device=device)
        
        # Create a NEW dictionary for the output state.
        state_out = {}

        # --- Input Layer ---
        in_state_in = state_in["in"] # Read from input state
        in_out, new_in_state = self.in_layer(ex_in + in_in, state=in_state_in)
        state_out["in"] = new_in_state # Write to output state

        # --- Hidden Layer (with internal steps) ---
        hid_state_in = state_in["hid"] # Read from input state
        hid_out = None
        
        # Note: We use hid_state_tensor to hold the state *within* the internal loop
        hid_state_tensor = hid_state_in 
        
        for _ in range(self.internal_steps):
            in2hid_influence = self.in2hid_IN(in_out)
            in2hid_gj_bundle = self.in2hid_GJ(in_out)
            
            hid_out, hid_state_tensor = self.hid_layer(
                in2hid_influence,
                state=hid_state_tensor, # Use most recent state
                gj_bundle=in2hid_gj_bundle,
                o_pre=in_out,
            )
            hid_ex_influence = self.hid_EX(hid_out)
            hid_in_influence = self.hid_IN(hid_out)
            hid_out, hid_state_tensor = self.hid_layer(
                hid_ex_influence + hid_in_influence,
                state=hid_state_tensor, # Use most recent state
            )

        state_out["hid"] = hid_state_tensor # Write final internal state

        # --- Output Layer ---
        hid2out_ex_influence = self.hid2out(hid_out)
        out_state_tensor, out_layer_state = self.out_layer(hid2out_ex_influence, state=state_in["out"])
        state_out["out"] = out_layer_state
        
        if self.log:
            self.log_monitor(state_out)
            
        return self.action_decoder(out_state_tensor), state_out

    # --- This is your original 'forward' method, renamed for evaluation ---
    def get_action_from_obs(self, x: torch.Tensor) -> torch.Tensor:
        """
        One TWC step for *evaluation/sampling*. (Stateful)
        This method uses and updates self._state internally.
        """
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
            
        ex_in, in_in = self.obs_encoder(x, n_inputs=4, device=device)
        B = ex_in.size(0)
        dtype = ex_in.dtype
        
        # Use and manage internal state
        state = self._ensure_state(B, device, dtype)
        
        # Detach the state for inference
        state = {name: (pair[0].detach(), pair[1].detach()) for name, pair in state.items()}

        in_state = state["in"]
        in_out, new_in_state = self.in_layer(ex_in + in_in, state=in_state)
        state["in"] = new_in_state # Inplace update of local 'state' dict

        hid_state = state["hid"]
        hid_out = None
        
        for _ in range(self.internal_steps):
            in2hid_influence = self.in2hid_IN(in_out)
            in2hid_gj_bundle = self.in2hid_GJ(in_out)
            
            hid_out, hid_state = self.hid_layer(
                in2hid_influence,
                state=hid_state,
                gj_bundle=in2hid_gj_bundle,
                o_pre=in_out,
            )
            hid_ex_influence = self.hid_EX(hid_out)
            hid_in_influence = self.hid_IN(hid_out)
            hid_out, hid_state = self.hid_layer(
                hid_ex_influence + hid_in_influence,
                state=hid_state,
            )
            state["hid"] = hid_state # Inplace update

        
        hid2out_ex_influence = self.hid2out(hid_out)
        out_state_tensor, out_layer_state = self.out_layer(hid2out_ex_influence, state=state["out"])
        state["out"] = out_layer_state
        
        self._state = state # Overwrite internal state
        
        if self.log:
            self.log_monitor(state)
            
        return self.action_decoder(out_state_tensor)

    # --- Helper methods for evaluation state ---
    def _make_state(self, batch_size: int, device, dtype):
        return self.get_initial_state(batch_size, device, dtype)

    def _ensure_state(self, batch_size: int, device, dtype):
        if self._state is None:
            self._state = self._make_state(batch_size, device, dtype)
            return self._state

        sample_E, _ = self._state["in"]
        if sample_E.shape[0] != batch_size or sample_E.device != device or sample_E.dtype != dtype:
            self._state = self._make_state(batch_size, device, dtype)
        return self._state
        
    def reset(self):
        """Resets the internal state variables for evaluation."""
        self._state = None
    
    # --- Other helpers ---
    def log_monitor(self, state):
        def _pack(layer, state_pair):
            return {
                "in_state": state_pair[0].detach().cpu(),
                "out_state": state_pair[1].detach().cpu(),
                "threshold": layer.threshold.detach().cpu(),
                "decay_factor": layer.decay.detach().cpu(),
            }
        self.monitor["in"].append(_pack(self.in_layer, state["in"]))
        self.monitor["hid"].append(_pack(self.hid_layer, state["hid"]))
        self.monitor["out"].append(_pack(self.out_layer, state["out"]))


# --- This is the builder function, now separate from the class ---

def create_layer(n_neurons) -> FIURIModule:
    return FIURIModule(
        num_cells=n_neurons,
        initial_in_state=0.0,
        initial_out_state=0.0,
        initial_threshold=0.0,
        initial_decay=0.1,
        clamp_min=-10.0,
        clamp_max=10.0,
    )

json_path = "TWC_fiu.json"

def build_twc(obs_encoder: Callable,
              action_decoder: Callable,
              internal_steps: int,
              log_stats: bool = True) -> nn.Module:
    """Builds and returns a TWC_BPTT model."""
    
    with open(json_path, "r") as f:
        net_data = json.load(f)

    masks, sizes = build_tw_matrices(net_data)

    n_in, n_hid, n_out = sizes["n_in"], sizes["n_hid"], sizes["n_out"]

    in_layer =  create_layer(n_in)
    hid_layer = create_layer(n_hid)
    out_layer = create_layer(n_out)

    in2hid = FiuriDenseConn(n_pre=n_in, n_post=n_hid,w_mask=masks["in2hid"]["IN"], type="IN")
    hid_IN = FiuriDenseConn(n_pre=n_hid, n_post=n_hid, w_mask=masks["hid"]["IN"], type="IN")
    hid_EX = FiuriDenseConn(n_pre=n_hid, n_post=n_hid, w_mask=masks["hid"]["EX"], type="EX")
    hid2out_EX = FiuriDenseConn(n_pre=n_hid, n_post=n_out, w_mask=masks["hid2out"]["EX"], type="EX")

    # create the only GJ sparse conn
    # PLM -> PVC, AVM -> AVD
    gj_edges = torch.tensor([[1, 2],   # src (PLM=1, AVM=2)
                             [2, 1]])  # dst (PVC=2, AVD=1)
    gj_conn = FiuriSparseGJConn(n_pre=n_in, n_post=n_hid, gj_edges=gj_edges)

    model = TWC_BPTT(
        in_layer=in_layer,
        hid_layer=hid_layer,
        out_layer=out_layer,
        in2hid_IN=in2hid,
        in2hid_GJ=gj_conn,
        hid_IN=hid_IN,
        hid_EX=hid_EX,
        hid2out_EX=hid2out_EX,
        obs_encoder=obs_encoder,
        action_decoder=action_decoder,
        internal_steps=internal_steps,
        log_stats=log_stats
    )
    return model

# %%
net = build_twc(twc.twc_io.mcc_obs_encoder_speed_weighted, twc.twc_io.twc_out_2_mcc_action_tanh, internal_steps=1)
print(net.state_dict())

# %%
class SequenceBuffer:
    """
    A replay buffer that stores and samples sequences of transitions for BPTT.
    
    This buffer stores entire episodes. When sampling, it pulls a fixed-length
    sequence from a random episode, ensuring that the sequence never
    crosses an episode boundary.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: The maximum number of *transitions* (not episodes) to store.
        """
        # Buffer of episodes. Each episode is a dict of numpy arrays.
        self.episodes = deque()
        
        self.capacity = int(capacity)
        self.total_transitions = 0
        
        # Temporary buffer for the episode currently being collected
        self._init_current_episode()

    @property
    def size(self) -> int:
        """Returns the total number of transitions stored in the buffer."""
        return self.total_transitions

    def _init_current_episode(self):
        """Resets the temporary episode buffer."""
        self.current_episode = {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "done": [],
        }

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, truncated: bool):
        """
        Stores a single transition. If 'done' or 'truncated' is True,
        the current episode is "flushed" to the main buffer.
        """
        # Ensure action is always at least 1D
        if not isinstance(action, np.ndarray):
             action = np.array([action])
        elif action.shape == (): # Handle 0-dim arrays
            action = np.array([action.item()])

        self.current_episode["obs"].append(obs)
        self.current_episode["action"].append(action)
        self.current_episode["reward"].append(np.array([reward])) # Store as (1,)
        self.current_episode["next_obs"].append(next_obs)
        self.current_episode["done"].append(np.array([done])) # Store as (1,)
        
        if done or truncated:
            self._flush_current_episode()

    def _flush_current_episode(self):
        ep_len = len(self.current_episode["obs"])
        if ep_len == 0:
            return

        flushed_episode = {}
        for key in self.current_episode.keys():
            flushed_episode[key] = np.stack(self.current_episode[key])
            
        self.episodes.append(flushed_episode)
        self.total_transitions += ep_len
        
        while self.total_transitions > self.capacity:
            evicted_episode = self.episodes.popleft()
            self.total_transitions -= len(evicted_episode["obs"])
            
        self._init_current_episode()

    def sample(self, batch_size: int, sequence_length: int, device: torch.device) -> dict:
        """
        Samples a batch of transition sequences for BPTT.
        """
        
        valid_episodes = [ep for ep in self.episodes if len(ep["obs"]) >= sequence_length]
        
        if not valid_episodes:
            raise ValueError(f"Not enough data to sample sequences. Need episodes >= {sequence_length} steps.")

        batch_seq = {key: [] for key in self.current_episode.keys()}

        for _ in range(batch_size):
            ep = random.choice(valid_episodes)
            max_start_idx = len(ep["obs"]) - sequence_length
            start = np.random.randint(0, max_start_idx + 1)
            end = start + sequence_length
            
            for key in batch_seq.keys():
                batch_seq[key].append(ep[key][start:end])

        tensor_batch = {}
        for key, data_list in batch_seq.items():
            stacked_data = np.stack(data_list)
            
            # --- FIX: Only flatten rewards and dones ---
            if key in ['reward', 'done'] and stacked_data.ndim == 3 and stacked_data.shape[2] == 1:
                stacked_data = stacked_data.reshape(batch_size, sequence_length)
                
            tensor_batch[key] = torch.tensor(stacked_data, dtype=torch.float32, device=device)

        return tensor_batch

# %%
from utils.ou_noise import OUNoise
from mlp import Critic
import itertools

# --- Hyperparameters (from twc_td3_bptt_train.py) ---
ENV = "MountainCarContinuous-v0"
SEED = 42
MAX_EPISODE        = 300
MAX_TIME_STEPS     = 999
WARMUP_STEPS       = 10_000 # Use this many steps to populate buffer
BATCH_SIZE         = 128
NUM_UPDATE_LOOPS   = 1 # Keep it simple for the notebook
POLICY_DELAY       = 2 # TD3 Policy Delay

GAMMA              = 0.99
TAU                = 5e-3
ACTOR_LR           = 3e-4 # Use a standard LR for this test
CRITIC_LR          = 3e-4

TWC_INTERNAL_STEPS = 1 # Keep it fast
CRITIC_HID_LAYERS  = [256, 256] # Smaller critics for faster test
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA_START, SIGMA_END, SIGMA_DECAY_EPIS = 0.20, 0.05, 100

# BPTT Hyperparameters
SEQUENCE_LENGTH    = 40  # Total length of sequences to sample
BURN_IN_LENGTH     = 10  # Number of steps to "warm up" the hidden state
TRAIN_LENGTH       = SEQUENCE_LENGTH - BURN_IN_LENGTH

# TD3 Noise Parameters
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP   = 0.5


# --- Helper Functions ---

def detach_state_dict(state_dict: dict):
    """Detaches all tensors in a state dictionary from the computation graph."""
    return {
        name: (state_pair[0].detach(), state_pair[1].detach())
        for name, state_pair in state_dict.items()
    }

def soft_update(target, source, tau):
    """Performs a soft update of target network parameters."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

@torch.no_grad()
def evaluate_policy(eval_env, actor_net, n_episodes=10):
    """Runs a policy evaluation loop."""
    actor_net.eval() # Set model to eval mode
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        actor_net.reset() # Reset the internal state for evaluation
        done = False
        while not done:
            # Use the stateful get_action_from_obs
            action_tensor = actor_net.get_action_from_obs(
                torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            )
            action = action_tensor.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
    actor_net.train() # Set model back to train mode
    return total_reward / n_episodes


# ---
# --- MAIN SCRIPT ---
# ---

print("--- Initializing ---")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Init Env and Buffer ---
env = gym.make(ENV)
eval_env = gym.make(ENV)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

buffer = SequenceBuffer(capacity=100_000)
ou_noise = OUNoise(action_dimension=act_dim, sigma=SIGMA_START)

# --- Init Networks ---
# Use the *real* encoders/decoders here
actor = build_twc(
    obs_encoder=twc.twc_io.mcc_obs_encoder_speed_weighted,
    action_decoder=twc.twc_io.twc_out_2_mcc_action_tanh,
    internal_steps=TWC_INTERNAL_STEPS,
    log_stats=False
).to(DEVICE)

critic_1 = Critic(obs_dim, act_dim, size=CRITIC_HID_LAYERS).to(DEVICE)
critic_2 = Critic(obs_dim, act_dim, size=CRITIC_HID_LAYERS).to(DEVICE)

# Create target networks
actor_target = build_twc(
    obs_encoder=twc.twc_io.mcc_obs_encoder_speed_weighted,
    action_decoder=twc.twc_io.twc_out_2_mcc_action_tanh,
    internal_steps=TWC_INTERNAL_STEPS,
    log_stats=False
).to(DEVICE)
critic_1_target = Critic(obs_dim, act_dim, size=CRITIC_HID_LAYERS).to(DEVICE)
critic_2_target = Critic(obs_dim, act_dim, size=CRITIC_HID_LAYERS).to(DEVICE)

# Initialize target networks
actor_target.load_state_dict(actor.state_dict())
critic_1_target.load_state_dict(critic_1.state_dict())
critic_2_target.load_state_dict(critic_2.state_dict())

# --- Init Optimizers ---
actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_opt = optim.Adam(itertools.chain(critic_1.parameters(), critic_2.parameters()), lr=CRITIC_LR)

print(f"Networks initialized. Running on {DEVICE}.")


# --- Populate Buffer ---
print(f"Populating buffer with {WARMUP_STEPS} random steps...")
obs, _ = env.reset()
actor.reset() # Reset stateful actor
for _ in range(WARMUP_STEPS):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    buffer.store(obs, action, reward, next_obs, terminated, truncated)
    obs = next_obs
    if terminated or truncated:
        obs, _ = env.reset()
        actor.reset()
print(f"Buffer populated with {buffer.size} transitions.")


# --- Main Training Loop ---
print("--- Starting Training Loop ---")
total_steps = WARMUP_STEPS
total_updates = 0

for ep in range(MAX_EPISODE):
    obs, _ = env.reset()
    actor.reset()
    ou_noise.update_sigma(ep)
    ou_noise.reset()
    
    ep_reward = 0
    ep_steps = 0
    
    for t in range(MAX_TIME_STEPS):
        # 1. Get Action (using stateful evaluation method)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_tensor = actor.get_action_from_obs(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            action = (action + ou_noise.noise()).clip(-max_action, max_action)

        # 2. Step Environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        ep_steps += 1
        total_steps += 1

        # 3. Store in Buffer
        buffer.store(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs

        # ---
        # --- BPTT Update Step ---
        # ---
        
        # We run the update logic (N times) *inside* the environment step loop
        for _ in range(NUM_UPDATE_LOOPS):
            try:
                # 1. Sample a batch of sequences
                batch_seq = buffer.sample(BATCH_SIZE, SEQUENCE_LENGTH, DEVICE)
                obs_seq = batch_seq["obs"]
                act_seq = batch_seq["action"]
                rew_seq = batch_seq["reward"]
                obs2_seq = batch_seq["next_obs"]
                term_seq = batch_seq["done"]
                
                # 2. Get initial hidden states (batched)
                actor_state = actor.get_initial_state(BATCH_SIZE, DEVICE)
                target_actor_state = actor_target.get_initial_state(BATCH_SIZE, DEVICE)

                # 3. Burn-in Loop (No Gradients)
                with torch.no_grad():
                    for i in range(BURN_IN_LENGTH):
                        _, actor_state = actor(obs_seq[:, i, :], actor_state)
                        _, target_actor_state = actor_target(obs2_seq[:, i, :], target_actor_state)

                # 4. Truncate (Detach)
                actor_state = detach_state_dict(actor_state)
                target_actor_state = detach_state_dict(target_actor_state)

                # 5. Training Loop (With Gradients)
                all_critic_losses = []
                all_actor_losses = []

                for i in range(BURN_IN_LENGTH, SEQUENCE_LENGTH):
                    # Get data for this time step
                    obs_t = obs_seq[:, i, :]
                    act_t = act_seq[:, i, :]
                    rew_t = rew_seq[:, i]
                    obs2_t = obs2_seq[:, i, :]
                    term_t = term_seq[:, i]
                    mask_t = 1.0 - term_t

                    # --- Compute Targets ---
                    with torch.no_grad():
                        noise = (torch.randn_like(act_t) * TARGET_POLICY_NOISE).clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
                        
                        next_a_t, target_actor_state = actor_target(obs2_t, target_actor_state)
                        next_a_t = (next_a_t + noise).clamp(-max_action, max_action)
                        
                        q1_t = critic_1_target(obs2_t, next_a_t)
                        q2_t = critic_2_target(obs2_t, next_a_t)
                        min_q = torch.minimum(q1_t, q2_t).squeeze(-1) # Squeeze Q-value
                        
                        q_target = rew_t + GAMMA * mask_t * min_q

                    # --- Compute Critic Loss ---
                    q1 = critic_1(obs_t, act_t).squeeze(-1) # Squeeze Q-value
                    q2 = critic_2(obs_t, act_t).squeeze(-1) # Squeeze Q-value
                    critic_loss_t = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
                    all_critic_losses.append(critic_loss_t)

                    # --- Compute Actor Loss (Delayed) ---
                    # We MUST unroll the actor to advance its state
                    a_t, actor_state = actor(obs_t, actor_state)
                    
                    if total_updates % POLICY_DELAY == 0:
                        actor_loss_t = -critic_1(obs_t, a_t).mean()
                        all_actor_losses.append(actor_loss_t)

                # 6. Backward Pass - Critic
                critic_loss = torch.stack(all_critic_losses).mean()
                
                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(itertools.chain(critic_1.parameters(), critic_2.parameters()), 1.0)
                critic_opt.step()

                # 7. Backward Pass - Actor (Delayed)
                total_updates += 1
                if total_updates % POLICY_DELAY == 0:
                    if all_actor_losses:
                        actor_loss = torch.stack(all_actor_losses).mean()
                        
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        actor_opt.step()
                        
                        # Soft update all target networks
                        soft_update(actor_target, actor, TAU)
                        soft_update(critic_1_target, critic_1, TAU)
                        soft_update(critic_2_target, critic_2, TAU)
                
            except ValueError as e:
                # Buffer might not be ready
                pass
        
        if done:
            break
            
    # --- End of Episode ---
    print(f"Ep: {ep+1} | Total Steps: {total_steps} | Reward: {ep_reward:.2f}")

    if (ep+1) % 10 == 0:
        eval_reward = evaluate_policy(eval_env, actor, n_episodes=10)
        print(f"----------------------------------------")
        print(f"Evaluation after {ep+1} episodes: {eval_reward:.2f}")
        print(f"----------------------------------------")

env.close()
eval_env.close()
print("--- Training Complete ---")

# %%



