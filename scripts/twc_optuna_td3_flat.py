import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import gymnasium as gym
import numpy as np
import torch
import optuna
import ast
import torch.nn.functional as F
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from tqdm import tqdm
from utils import OUNoise, SequenceBuffer
from mlp import TwinCritic
from fiuri import PyUriTwc_V2, build_fiuri_twc_v2

@dataclass
class TD3Config:
    # --- Training loop ---
    max_train_steps: int = 300_000
    warmup_steps: int = 10_000
    batch_size: int = 256  # Standard stable batch size for MCC
    num_update_loops: int = 1 # Updates per step (Standard is 1)
    update_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # --- Evaluation ---
    eval_interval_episodes: int = 10
    eval_episodes: int = 10

    # --- BPTT options ---
    sequence_length: int = 8
    burn_in_length: int = 4

    # Hyperparameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    
    exp_noise: float = 0.1 # most common fixed std in TD3 algorithms
    # OU Noise Parameters
    ou_theta: float = 0.15
    ou_sigma_init: float = 0.5 # Higher initial noise for exploration
    ou_sigma_end: float = 0.1
    ou_sigma_decay_steps: int = 100_000 # Decays over first 1/3 of training
    
    # SG version hyperparameters
    steepness_fire: float = 14.0
    steepness_gj: float = 7.0
    steepness_input: float = 5.0
    input_thresh: float = 0.001

    critic_hidden_layers: int = 256    
    replay_buffer_size: int = 100_000
    
    model_prefix: str = "td3_flat_actor"
    
    def to_json(self) -> str:
        d = asdict(self)
        d["critic_hidden_layers"] = str(self.critic_hidden_layers)
        return json.dumps(d, indent=4)

    def load(self, json_data):
        data = json.loads(json_data) if isinstance(json_data, str) else dict(json_data)
        
        # Handle tuple fields
        layers = data.get("critic_hidden_layers")
        if isinstance(layers, str):
            try:
                layers = json.loads(layers)
            except:
                layers = ast.literal_eval(layers)
        if isinstance(layers, (list, tuple)):
            data["critic_hidden_layers"] = tuple(layers)

        for field in self.__dataclass_fields__:
            if field in data:
                setattr(self, field, data[field])
        return self
    
class TD3Engine():
    """  
    TODO: UPDATE DOCSTRING
    
    """
    def __init__(self,
                 gamma: float,
                 tau: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 actor: PyUriTwc_V2,
                 critic: TwinCritic,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        
        self.gamma = gamma
        self.tau = tau
        self.obs_space = observation_space
        self.act_space = action_space
        self.actor = actor
        self.critic = critic
        self.actor_target = deepcopy(actor)
        self.critic_target = deepcopy(critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer  # should include both critics' params
        self.device = device

        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)

        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)

        # cache bounds as tensors
        self.action_low  = torch.as_tensor(self.act_space.low,  device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.act_space.high, device=self.device, dtype=torch.float32)

        self.total_updates = 0  # for delayed actor
    
    @torch.no_grad()
    def soft_update(self, target_net, online_net, tau):
        # 1. Soft Update Learnable Parameters (Weights/Biases)
        for target_param, param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        # 2. Hard Copy Non-Learnable Buffers (BatchNorm stats)
        # Buffers (running_mean, running_var) are not "parameters" so they are skipped above.
        for target_buffer, buffer in zip(target_net.buffers(), online_net.buffers()):
            target_buffer.data.copy_(buffer.data)
                
    def get_action(self, state):
        # Shape: (1, ObsDim)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            a = self.actor(s) # Expected (1, ActionDim)

        # Clamp using broadcasted shapes, then squeeze to (ActionDim,)
        return torch.clamp(a, self.action_low, self.action_high).squeeze(0)

    def _detach_state_tuple(self, state_tuple):
        """Helper to detach (E, O) tuple from TWC_V2"""
        return (state_tuple[0].detach(), state_tuple[1].detach())
    
    def update_step_bptt(self, 
                         batch: dict, 
                         burn_in: int,):
        obs = batch['obs'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        terminated = batch['terminated'].to(self.device)
        truncated = batch['truncated'].to(self.device)

        B = obs.shape[0]

        # 1. Burn-in
        with torch.no_grad():
            init_E, init_O = self.actor.get_initial_state(B, self.device)
            if burn_in > 0:
                _, (h_E, h_O) = self.actor.forward_bptt(obs[:, :burn_in], (init_E, init_O))
                _, (h_E_t, h_O_t) = self.actor_target.forward_bptt(next_obs[:, :burn_in], (init_E, init_O))
            else:
                h_E, h_O = init_E, init_O
                h_E_t, h_O_t = init_E, init_O
            
        # Training Sequence Slices
        obs_t = obs[:, burn_in:]
        next_obs_t = next_obs[:, burn_in:]
        act_t = action[:, burn_in:]
        rew_t = reward[:, burn_in:].unsqueeze(-1)
        terminated_t = terminated[:, burn_in:].unsqueeze(-1)
        truncated_t = truncated[:, burn_in:].unsqueeze(-1)
        done_t = terminated_t 

        # 2. Critic Update
        with torch.no_grad():
            # Get next action from target actor
            next_act, _ = self.actor_target.forward_bptt(next_obs_t, (h_E_t, h_O_t))
            
            # Add noise
            noise = (torch.randn_like(next_act) * self.target_policy_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_act = (next_act + noise).clamp(self.action_low, self.action_high)            
            # Target Q
            q1_t, q2_t = self.critic_target(next_obs_t, next_act)
            min_q = torch.min(q1_t, q2_t)            
            target_q = rew_t + (1 - done_t) * self.gamma * min_q

        q1, q2 = self.critic(obs_t, act_t)        
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 3. Actor Update
        actor_loss_val = 0.0
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:

            h_E_detached = h_E.detach()
            h_O_detached = h_O.detach()

            pi_act, _ = self.actor.forward_bptt(obs_t, (h_E_detached, h_O_detached))
            actor_loss = -self.critic.q1_forward(obs_t, pi_act).mean()
            
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update targets
            self.soft_update(target_net=self.actor_target, 
                             online_net=self.actor, 
                             tau=self.tau)
            self.soft_update(target_net=self.critic_target, 
                             online_net=self.critic, 
                             tau=self.tau)
            
            actor_loss_val = actor_loss.item()
            
        return actor_loss_val, critic_loss.item(), 
    
    @torch.no_grad()
    def evaluate_policy(self, env, episodes=10):
        self.actor.eval()
        total = 0.0
        eval_actions = []
        for _ in range(episodes):
            obs, _ = env.reset()
            self.actor.reset()
            done = False
            while not done:
                a = self.get_action(obs)
                a_np = a.detach().cpu().numpy()
                eval_actions.append(a_np)
                obs, r, terminated, truncated, _ = env.step(a_np)
                total += r
                done = terminated or truncated
        self.actor.train()
        return (total / episodes), np.mean(eval_actions)


def td3_train(
    env: gym.Env,
    replay_buf: SequenceBuffer,
    engine: TD3Engine,
    writer: SummaryWriter,
    timestamp: str,
    config: TD3Config,
    trial: optuna.Trial,
    OUNoise: OUNoise,
):
    """
    Main training loop for TD3, adapted to run for a maximum number of time steps.
    """
    total_steps = 0
    best_ret = -np.inf
    best_model_path = None
    e = 0  # Episode counter

    # loop variables
    env_seed = config.seed
    max_train_steps = config.max_train_steps
    warmup_steps =  config.warmup_steps
    num_update_loops = config.num_update_loops
    update_every_steps = config.update_every
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    device = config.device
    burn_in_length = config.burn_in_length
    eval_interval_episodes =  config.eval_interval_episodes
    eval_episodes =  config.eval_episodes
    model_prefix = config.model_prefix
    exp_noise = config.exp_noise
    action_low  = torch.as_tensor(engine.act_space.low,  device=engine.device, dtype=torch.float32)
    action_high = torch.as_tensor(engine.act_space.high, device=engine.device, dtype=torch.float32)
    
    print(f"working on device: {device}")
    # Use tqdm to track total steps
    pbar = tqdm(total=config.max_train_steps, initial=total_steps, desc="Training TD3")

    eval_idx = 0
    all_eval_scores = []
    pruning_window = deque(maxlen=5)

    while total_steps < max_train_steps:
        # New episode
        obs, _ = env.reset(seed=env_seed)
        ep_reward = 0.0
        ep_actions = []
        steps = 0

        OUNoise.reset()

        engine.actor.reset()
        engine.actor_target.reset()
        
        done = False

        while not done:
            if total_steps >= max_train_steps:
                break
            # Action Selection
            a_det = engine.get_action(obs)

            if total_steps < warmup_steps: 
                action = env.action_space.sample()
            else: 
                """  
                this get action, makes a forward pass where the actor maintain his own state
                through all the active training episode. In the update step network internal state
                is managed differntly to avoid intervinig in the state of the network during
                the active episode. 
                """                
                OUNoise.update(total_steps=total_steps)
                noise = torch.as_tensor(OUNoise.noise(), 
                                            device=a_det.device,
                                            dtype=a_det.dtype)

                action = torch.clamp(a_det + noise, action_low, action_high).detach().cpu().numpy()

            # Environment step 
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_actions.append(action[0])
            
            # Store transition and update step counters 
            replay_buf.store(obs, action, reward, obs2, terminated, truncated)
            obs = obs2
            total_steps += 1
            steps += 1
            
            # Update every X episodes 
            if total_steps > warmup_steps and (total_steps % update_every_steps == 0):
                for _ in range(num_update_loops):
                    seq_batch = replay_buf.sample(batch_size, sequence_length, device)
                    actor_loss, critic_loss = engine.update_step_bptt(seq_batch, burn_in_length)

                # Log losses every 100 steps to avoid exessive IO
                if total_steps % 100 == 0:
                    writer.add_scalar('Loss/Actor', actor_loss, total_steps)
                    writer.add_scalar('Loss/Critic', critic_loss, total_steps)
                    if OUNoise:
                        writer.add_scalar('Training/OUNoise_sigma', OUNoise.sigma, total_steps)
        
        pbar.update(steps) # Update the tqdm progress bar
        
        if done:
            # Current ep ended, log middle trainig results
            writer.add_scalar('Training/Episode_Return', ep_reward, total_steps)
            writer.add_scalar('Training/Episode_steps', steps, total_steps)
            if len(ep_actions) > 0:
                writer.add_scalar('Training/AvgAction', float(np.mean(ep_actions)), total_steps)
                writer.add_scalar('Training/StdAction', float(np.std(ep_actions)), total_steps)
            
            e += 1

            # Evaluation & Optuna Pruning
            if e % eval_interval_episodes == 0:
                eval_ret, eval_avg_action = engine.evaluate_policy(env, episodes=eval_episodes)
                writer.add_scalar('Evaluation/Return', eval_ret, total_steps)
                writer.add_scalar('Evaluation/AvgAction', eval_avg_action, total_steps)

                tqdm.write(f"\nEpisode {e}: TotalSteps: {total_steps}, EvalReturn: {eval_ret:.2f}")
                all_eval_scores.append(eval_ret)
                pruning_window.append(eval_ret)
                rolling_avg = np.mean(pruning_window)
                if eval_ret > best_ret:
                    best_ret = eval_ret
                    prefix = model_prefix
                    model_path = os.path.join(writer.log_dir, f"{prefix}_best_{timestamp}.pth")
                    torch.save(engine.actor.state_dict(), model_path)
                    best_model_path = model_path
                    tqdm.write(f"New best evaluation reward: {best_ret:.2f}. Model saved to {model_path}")
                
                # --- OPTUNA PRUNING LOGIC ---
                if trial is not None:
                    trial.report(rolling_avg, step=eval_idx)
                    if trial.should_prune():
                        tqdm.write(f"Trial {trial.number} pruned at eval {eval_idx} (episode {e}) with best return {best_ret:.2f}.")
                        pbar.close()
                        raise optuna.TrialPruned()
                eval_idx += 1


    pbar.close()
    
    # --- Final Save ---
    prefix = f"{model_prefix}_final"
    model_path = os.path.join(writer.log_dir, f"{prefix}_{timestamp}.pth")            
    torch.save(engine.actor.state_dict(), model_path)
    print(f"Final Model saved to {model_path}")           
    return all_eval_scores, best_model_path


def make_env(seed, env_id="MountainCarContinuous-v0"):
    import gymnasium as gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def objective(trial: optuna.Trial, study_name: str):
    cfg = TD3Config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shorter runs for search; keep evaluation cadence stable
    cfg.max_train_steps = 250_000
    cfg.warmup_steps = 10_000
    cfg.eval_interval_episodes = 10
    cfg.eval_episodes = 10
    cfg.num_update_loops = 1
    cfg.update_every = 1
    cfg.batch_size = 256
    cfg.model_prefix = "td3_flat_actor"

    # --- Tunable Hyperparameters ---
    cfg.actor_lr = trial.suggest_float("actor_lr", 1.5e-4, 4.0e-4, log=True)
    cfg.critic_lr = trial.suggest_float("critic_lr", 1.5e-4, 4.0e-4, log=True)
    cfg.gamma = trial.suggest_float("gamma", 0.978, 0.993)
    cfg.tau = trial.suggest_float("tau", 5e-3, 1.2e-2)
    cfg.target_noise = trial.suggest_float("target_noise", 0.20, 0.36)
    cfg.noise_clip = trial.suggest_float("noise_clip", 0.25, 0.45)
    cfg.ou_sigma_init = trial.suggest_float("sigma_start", 0.30, 0.50)
    cfg.ou_sigma_end = trial.suggest_float("sigma_end", 0.05, 0.12)
    cfg.steepness_fire = trial.suggest_float("steepness_fire", 12.0, 16.0)
    cfg.steepness_gj = trial.suggest_float("steepness_gj", 6.0, 9.5)
    cfg.steepness_input = trial.suggest_float("steepness_input", 4.0, 6.5)
    cfg.input_thresh = trial.suggest_float("input_thresh", 8e-4, 2e-3, log=True)

    # Per-trial seed to reduce correlation across samples
    seed = cfg.seed + trial.number
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = build_fiuri_twc_v2(
        steepness_gj=cfg.steepness_gj,
        steepness_fire=cfg.steepness_fire,
        steepness_input=cfg.steepness_input,
        input_thresh=cfg.input_thresh,
    )
    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    engine = TD3Engine(
        gamma=cfg.gamma,
        tau=cfg.tau,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        policy_delay=cfg.policy_delay,
        target_policy_noise=cfg.target_noise,
        target_noise_clip=cfg.noise_clip,
        device=cfg.device,
    )

    replay_buf = SequenceBuffer(capacity=cfg.replay_buffer_size)

    cfg.ou_sigma_decay_steps = int(cfg.max_train_steps * 0.7)
    noise = OUNoise(
        size=env.action_space.shape,
        mu=0.0,
        theta=0.15,
        sigma_init=cfg.ou_sigma_init,
        sigma_min=cfg.ou_sigma_end,
        decay_steps=cfg.ou_sigma_decay_steps,
        dt=1.0,
        seed=seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"trial_{trial.number}_{timestamp}"
    log_dir = f"out/runs/optuna/{study_name}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    with open(os.path.join(log_dir, "trial_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)

    with open(os.path.join(log_dir, "full_config.json"), "w") as f:
        f.write(cfg.to_json())

    try:
        all_eval_scores, best_model_path = td3_train(
            env=env,
            replay_buf=replay_buf,
            engine=engine,
            writer=writer,
            timestamp=timestamp,
            config=cfg,
            trial=trial,
            OUNoise=noise,
        )

        final_stability_score = np.mean(all_eval_scores[-5:]) 
        return final_stability_score
    except optuna.TrialPruned:
        return -np.inf
    finally:
        env.close()
        writer.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = "twc_mcc_td3_flat_optuna"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )

    sampler = TPESampler(
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///db.sqlite3",
        study_name=study_name,
        load_if_exists=True,
    )

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(study.trials) == 0:
        baseline = {
            "actor_lr": 0.00024319130600441337,
            "critic_lr": 0.00028664900582161976,
            "gamma": 0.9922798944951167,
            "tau": 0.006443939133870398,
            "target_noise": 0.35757304576778454,
            "noise_clip": 0.3919642928588686,
            "sigma_start": 0.3244395391061171,
            "sigma_end": 0.11638525345225356,
            "steepness_fire": 14.188993516206018,
            "steepness_gj": 7.3502474066661225,
            "steepness_input": 6.146129878780117,
            "input_thresh": 0.000932138862222648
        }
        study.enqueue_trial(baseline)
    elif len(completed_trials) > 0:
        best_params = study.best_trial.params
        print("\nEnqueuing previous best params for re-evaluation...")
        study.enqueue_trial(best_params)

    obj_wrapper = lambda trial: objective(trial, study_name)

    try:
        study.optimize(
            obj_wrapper,
            n_trials=20,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed_trials:
        print(f"\nStudy '{study_name}' complete.")
        print(f"Best value (max eval return): {study.best_value:.2f}")
        print("Best params:")
        print(json.dumps(study.best_params, indent=4))

        out_dir = f"out/runs/optuna/{study_name}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        best_params_path = os.path.join(out_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print(f"Saved to {best_params_path}")
    else:
        print(f"\nStudy '{study_name}' has no completed trials yet (still running in other workers?).")


if __name__ == "__main__":
    main()
