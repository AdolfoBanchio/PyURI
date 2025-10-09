from collections import defaultdict
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.twc_builder import build_twc
from utils.twc_io_wrapper import mountaincar_pair_encoder

device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# HYPERPARAMETERS AND DATA COLLECTION PARAMS
lr = 5e-3
max_grad_norm = 1.0

frames_per_batch = 1024
total_frames = 10_240*10

sub_batch_size = 256  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 2e-2  # encourage more exploration to escape plateaus

base_env = GymEnv("MountainCarContinuous-v0", device=device)

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

check_env_specs(env)

rollout = env.rollout(3)

twc_net = build_twc(action_decoder=mountaincar_pair_encoder(), use_json_w=True)

actor_net = nn.Sequential(
    twc_net,
    nn.LayerNorm(2),  # stabilize TWC outputs before param extraction
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc","scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc","scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec_unbatched.space.low,
        "high": env.action_spec_unbatched.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_net = nn.Sequential(
    nn.LayerNorm(2),
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
    )

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)


collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coeff=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

logs = defaultdict(list)
# TensorBoard writer (timestamped run directory)
run_name = f"twc_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # Ensure no computation graph from collection is kept
    tensordict_data = tensordict_data.detach()
    # we now have a batch of data to work with. Let's learn something from it.
    for e in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        # Cut any new graphs introduced by value recomputation
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            # Recurrent state should not backprop across sub-batches
            tensordict_data = tensordict_data.detach()
            twc_net._detach()
            subdata = replay_buffer.sample(sub_batch_size).detach()
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            # Also detach recurrent states after each update to avoid stale graphs
            optim.zero_grad()

    # Basic metrics per batch
    batch_avg_rew = tensordict_data["next", "reward"].mean().item()
    logs["reward"].append(batch_avg_rew)
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    logs["lr"].append(optim.param_groups[0]["lr"])

    # TensorBoard logging (use global steps as frames seen)
    global_steps = pbar.n + tensordict_data.numel()
    writer.add_scalar("train/avg_reward", batch_avg_rew, global_steps)
    writer.add_scalar("train/step_count_max", logs["step_count"][-1], global_steps)
    writer.add_scalar("train/lr", logs["lr"][-1], global_steps)

    # Progress bar update and concise status
    pbar.update(tensordict_data.numel())
    cum_reward_str = f"avg_rew={logs['reward'][-1]:.3f}"
    lr_str = f"lr={logs['lr'][-1]:.2e}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            # TensorBoard eval
            writer.add_scalar("eval/reward_mean", logs["eval reward"][-1], pbar.n)
            writer.add_scalar("eval/reward_sum", logs["eval reward (sum)"][-1], pbar.n)
            writer.add_scalar("eval/step_count_max", logs["eval step_count"][-1], pbar.n)
            eval_str = f"eval_sum={logs['eval reward (sum)'][-1]:.3f}"
            del eval_rollout
    # Minimal pbar description
    pbar.set_description(", ".join([s for s in [eval_str, cum_reward_str, lr_str] if s]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

# Close writer and export CSV logs
pbar.close()
writer.flush(); writer.close()

csv_path = f"logs_{run_name}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    # header
    header = [
        "idx",
        "train_avg_reward",
        "train_step_count_max",
        "lr",
        "eval_reward_mean",
        "eval_reward_sum",
        "eval_step_count_max",
    ]
    w.writerow(header)
    # rows (pad shorter eval lists with NaN)
    N = len(logs["reward"])
    for idx in range(N):
        row = [
            idx,
            logs["reward"][idx] if idx < len(logs["reward"]) else float("nan"),
            logs["step_count"][idx] if idx < len(logs["step_count"]) else float("nan"),
            logs["lr"][idx] if idx < len(logs["lr"]) else float("nan"),
            logs["eval reward"][idx] if idx < len(logs.get("eval reward", [])) else float("nan"),
            logs["eval reward (sum)"][idx] if idx < len(logs.get("eval reward (sum)", [])) else float("nan"),
            logs["eval step_count"][idx] if idx < len(logs.get("eval step_count", [])) else float("nan"),
        ]
        w.writerow(row)

print(f"TensorBoard logs: runs/{run_name} | CSV: {csv_path}")

