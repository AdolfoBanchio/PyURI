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
from optuna.trial import TrialState
from tqdm import tqdm
from utils import OUNoise, SequenceBuffer
from mlp import TwinCritic
from fiuri import PyUriTwc

@dataclass
class TD3Config:
    # --- Training loop ---
    max_train_steps: int = 300_000
    warmup_steps: int = 10_000
    batch_size: int = 256
    num_update_loops: int = 1
    update_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # --- Evaluation ---
    eval_interval_episodes: int = 10
    eval_episodes: int = 100

    # --- BPTT options ---
    sequence_length: int = 32
    burn_in_length: int = 16

    # Hyperparameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5

    # OU Noise Parameters
    ou_theta: float = 0.15
    ou_sigma_init: float = 0.5
    ou_sigma_end: float = 0.1
    ou_sigma_decay_steps: int = 100_000

    critic_hidden_layers: int = 256
    replay_buffer_size: int = 100_000

    # SG version hyperparameters
    steepness_fire: float = 14.0
    steepness_gj: float = 7.0
    steepness_input: float = 5.0
    input_thresh: float = 0.001
    
    model_prefix: str = "td3_flat_actor"

    def to_json(self) -> str:
        d = asdict(self)
        d["critic_hidden_layers"] = str(self.critic_hidden_layers)
        d["device"] = str(self.device)
        return json.dumps(d, indent=4)

    def load(self, json_data):
        data = json.loads(json_data) if isinstance(json_data, str) else dict(json_data)

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

# ------------------------------------------------------------
# TD3 Engine
# ------------------------------------------------------------

class TD3Engine:
    def __init__(
        self,
        gamma: float,
        tau: float,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor: PyUriTwc,
        critic: TwinCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.obs_space = observation_space
        self.act_space = action_space
        self.actor = actor
        self.critic = critic
        self.actor_target = deepcopy(actor)
        self.critic_target = deepcopy(critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.device = device

        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)

        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)

        self.action_low = torch.as_tensor(self.act_space.low, device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.act_space.high, device=self.device, dtype=torch.float32)

        self.total_updates = 0

    @torch.no_grad()
    def soft_update(self, target_net, online_net):
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        for target_buffer, buffer in zip(target_net.buffers(), online_net.buffers()):
            target_buffer.data.copy_(buffer.data)

    def get_action(self, state, noise=0.0):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state) # Llama al forward() con memoria
        if noise != 0:
            action = action + torch.randn_like(action) * noise
        action_np = action.detach().cpu().numpy()[0]
        return np.clip(action_np, self.act_space.low, self.act_space.high)

    def update_step_bptt(self, batch: dict, burn_in: int):
        obs = batch['obs']        # (B, T, 2)
        action = batch['action']   # (B, T, 1)
        reward = batch['reward']   # (B, T, 1)
        next_obs = batch['next_obs']
        not_done = 1.0 - batch['terminated']

        B = obs.shape[0]

        with torch.no_grad():
            init_E, init_O = self.actor.get_initial_state(B, self.device)
            if burn_in > 0:
                _, (h_E, h_O) = self.actor.forward_bptt(obs[:, :burn_in], (init_E, init_O))
                _, (h_E_t, h_O_t) = self.actor_target.forward_bptt(next_obs[:, :burn_in], (init_E, init_O))
            else:
                h_E, h_O = init_E, init_O
                h_E_t, h_O_t = init_E, init_O


        obs_t = obs[:, burn_in:]
        action_t = action[:, burn_in:]
        reward_t = reward[:, burn_in:]
        next_obs_t = next_obs[:, burn_in:]
        terminated_t = batch['terminated'][:, burn_in:]

        done_t = terminated_t
        with torch.no_grad():
            next_act, _ = self.actor_target.forward_bptt(next_obs_t, (h_E_t, h_O_t))

            # Add noise
            noise = (torch.randn_like(next_act) * self.target_policy_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_act = (next_act + noise).clamp(self.action_low, self.action_high)            
            # Target Q
            q1_t, q2_t = self.critic_target(next_obs_t, next_act)
            min_q = torch.min(q1_t, q2_t)            
            target_q = reward_t + (1 - done_t) * self.gamma * min_q

        q1, q2 = self.critic(obs_t, action_t)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor update
        actor_loss_val = 0.0
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            h_E_det, h_O_det = h_E.detach(), h_O.detach()
            
            action, _ = self.actor.forward_bptt(obs_t, (h_E_det, h_O_det))
            actor_loss = -self.critic.q1_forward(obs_t, action).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

            actor_loss_val = actor_loss.item()
        
        return actor_loss_val, critic_loss.item()

    @torch.no_grad()
    def evaluate_policy(self, env, episodes: int):
        """
        Evaluación extendida:
        - mean_return
        - success_rate (terminated=True)
        - avg_steps_success, median_steps_success (solo éxitos)
        - mean_action, mean_abs_action
        - pct_near_zero_action: fracción de acciones con |a| < eps
        """
        self.actor.eval()

        returns = []
        successes = []
        steps_list = []
        steps_success = []
        all_actions = []

        for _ in range(episodes):
            obs, _ = env.reset()
            self.actor.reset()

            done = False
            ep_ret = 0.0
            ep_steps = 0
            ep_success = 0

            while not done:
                a_np = self.get_action(obs)
                all_actions.append(float(a_np[0]) if np.ndim(a_np) > 0 else float(a_np))

                obs, r, terminated, truncated, _ = env.step(a_np)
                ep_ret += float(r)
                ep_steps += 1
                done = bool(terminated or truncated)

                if terminated:
                    ep_success = 1

            returns.append(ep_ret)
            successes.append(ep_success)
            steps_list.append(ep_steps)
            if ep_success:
                steps_success.append(ep_steps)

        actions = np.array(all_actions, dtype=np.float32) if len(all_actions) else np.array([0.0], dtype=np.float32)

        mean_return = float(np.mean(returns)) if returns else 0.0
        success_rate = float(np.mean(successes)) if successes else 0.0

        if len(steps_success) > 0:
            avg_steps_success = float(np.mean(steps_success))
        else:
            avg_steps_success = float("inf")

        mean_action = float(np.mean(actions))

        self.actor.train()

        metrics = {
            "mean_return": mean_return,
            "success_rate": success_rate,
            "avg_steps_success": avg_steps_success,
            "mean_action": mean_action,
        }
        return metrics


# ------------------------------------------------------------
# Helpers: shaping and eval scoring
# ------------------------------------------------------------

def phi(obs):
    # Potential shaping (pos only). Keeps your original behavior.
    pos = obs[0]
    pos_min, pos_max = -1.2, 0.6
    x = (pos - pos_min) / (pos_max - pos_min)
    return 4 * float(np.clip(x, 0.0, 1.0))

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

def td3_train(
    env: gym.Env,
    replay_buf: SequenceBuffer,
    engine: TD3Engine,
    writer: SummaryWriter,
    timestamp: str,
    config: TD3Config,
    trial=None,
):
    total_steps = 0
    e = 0  # episodes
    best_combined_score = -np.inf
    best_model_path = None
    best_model_path_by_ret = None

    pbar = tqdm(total=config.max_train_steps, desc="Training TD3")

    eval_idx = 0
    pruning_window = deque(maxlen=6)
    all_eval_scores = []
    all_eval_steps = []
    ou_noise = OUNoise(env.action_space.shape[0], config)
    # initial model save
    initial_path = os.path.join(writer.log_dir, f"{config.model_prefix}_first_{timestamp}.pth")
    torch.save(engine.actor.state_dict(), initial_path)
    print(f"Initial Model saved to {initial_path}")

    # Training loop
    while total_steps < config.max_train_steps:
        obs, _ = env.reset(seed=config.seed + e)
        ou_noise.reset()
        engine.actor.reset()

        ep_reward = 0.0
        ep_actions = []
        steps = 0
        done = False
        ep_terminated = False

        while not done:
            if total_steps >= config.max_train_steps:
                break

            # 1. Selección de Acción
            if total_steps < config.warmup_steps:
                action = env.action_space.sample()
            else:
                # get_action ya devuelve numpy clipeado
                a_det = engine.get_action(obs)
                ou_noise.update(total_steps)
                noise = ou_noise.noise()
                action = np.clip(a_det + noise, env.action_space.low, env.action_space.high)

            obs2, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            if terminated:
                ep_terminated = True

            ep_reward += float(reward)
            ep_actions.append(float(action[0]) if np.ndim(action) > 0 else float(action))

            # Potential-based shaping (usa engine.gamma)
            shaped = float(reward + engine.gamma * phi(obs2) - phi(obs))
            replay_buf.store(obs, action, shaped, obs2, terminated, truncated)
            obs = obs2
            total_steps += 1
            steps += 1

            if total_steps > config.warmup_steps and (total_steps % config.update_every == 0):
                if replay_buf.total_transitions > config.batch_size * 2: 
                    for _ in range(config.num_update_loops):
                        seq_batch = replay_buf.sample(config.batch_size, config.sequence_length)
                        if seq_batch is not None:
                            actor_loss, critic_loss = engine.update_step_bptt(seq_batch, config.burn_in_length)
                
                if total_steps % 100 == 0:
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Training/OUNoise_sigma", ou_noise.sigma, total_steps)

        pbar.update(steps)

        # episode ended
        writer.add_scalar("Training/Episode_Return", ep_reward, total_steps)
        writer.add_scalar("Training/Episode_steps", steps, total_steps)
        if len(ep_actions) > 0:
            ep_actions_np = np.array(ep_actions, dtype=np.float32)
            writer.add_scalar("Training/AvgAction", float(np.mean(ep_actions_np)), total_steps)
            writer.add_scalar("Training/StdAction", float(np.std(ep_actions_np)), total_steps)

        e += 1

        # evaluation
        if e % config.eval_interval_episodes == 0:
            metrics = engine.evaluate_policy(env, episodes=config.eval_episodes)

            eval_ret = metrics["mean_return"]
            success_rate = metrics["success_rate"]
            avg_steps_success = metrics["avg_steps_success"]
            mean_action = metrics["mean_action"]

            # log metrics
            writer.add_scalar("Evaluation/Return", eval_ret, total_steps)
            writer.add_scalar("Evaluation/SuccessRate", success_rate, total_steps)

            # steps can be inf: log a capped value to keep TB sane
            writer.add_scalar(
                "Evaluation/AvgStepsSuccess",
                float(avg_steps_success if np.isfinite(avg_steps_success) else 1e3),
                total_steps,
            )

            writer.add_scalar("Evaluation/MeanAction", mean_action, total_steps)

            tqdm.write(
                f"\nEpisode {e} | Steps {total_steps} | "
                f"Ret {eval_ret:.2f} | Succ {100*success_rate:.1f}% | "
                f"AvgStepsSucc {avg_steps_success if np.isfinite(avg_steps_success) else -1:.1f} | "
            )

            all_eval_scores.append(eval_ret)
            all_eval_steps.append(avg_steps_success)


             
            # Calculamos las medias de la ventana (6)
            window_rew = np.mean(all_eval_scores[-6:])
            # Si avg_steps es inf (no llegó), le asignamos el máximo del entorno (1000)
            clean_steps = [s if np.isfinite(s) else 1000 for s in all_eval_steps[-6:]]
            window_steps = np.mean(clean_steps)

            step_penalty = max(0, (window_steps - 200) * 0.1)
            current_combined_score = window_rew - step_penalty

            # best 
            if current_combined_score > best_combined_score:
                best_combined_score = current_combined_score
                path = os.path.join(writer.log_dir, f"{config.model_prefix}_best_{timestamp}.pth")
                torch.save(engine.actor.state_dict(), path)
                best_model_path = path
                tqdm.write(f"New best RETURN: {eval_ret:.2f}. Model saved to {path}")

            # Optuna pruning: report score (no return)
            if trial is not None:
                trial.report(best_combined_score, step=eval_idx+1)
                if trial.should_prune():
                    tqdm.write(
                        f"Trial {trial.number} pruned at eval {eval_idx} "
                        f"(episode {e}) best_score={current_combined_score:.1f} rolling={window_rew:.1f}"
                    )
                    pbar.close()
                    raise optuna.TrialPruned()

            eval_idx += 1

    pbar.close()

    # Final save
    final_path = os.path.join(writer.log_dir, f"{config.model_prefix}_final_{timestamp}.pth")
    torch.save(engine.actor.state_dict(), final_path)
    print(f"Final Model saved to {final_path}")

    # Return eval scores and best path (by score)
    return all_eval_scores, all_eval_steps, best_model_path
