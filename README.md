Tools used:
1. dnSpy

rollout section — describes the experience collected this iteration:

ep_len_mean: average number of steps per episode.
ep_rew_mean: average total reward per episode. The most important metric for tracking whether your agent is improving.

time section — bookkeeping:
fps: steps per second. Ours is 0-1 because each Jump King step takes several real seconds.
iterations: how many n_steps buffers have been collected so far.
time_elapsed: seconds since training started.
total_timesteps: total steps taken this session.

train section — describes the neural network update:

approx_kl: how much the policy changed during this update. Too high means unstable updates, too low means barely learning. Ideally stays small but nonzero.
clip_fraction: fraction of updates that hit PPO's clipping threshold. Near 0 means the policy is barely changing — could indicate the learning rate is too low or the policy has stagnated.
clip_range: the clipping threshold itself, fixed at 0.2 by default.
entropy_loss: measures how random the policy is. High magnitude means more exploratory, decreasing over time means the policy is becoming more confident.
explained_variance: how well the value function predicts actual returns. Near 0 means it's essentially guessing. Near 1 means it's accurately predicting future rewards — this is what you want.
learning_rate: your current learning rate, fixed at 0.0003.
loss: combined loss of the policy and value function. Variable and hard to interpret in isolation.
n_updates: cumulative number of gradient updates applied to the network.
policy_gradient_loss: how much the policy is being pushed to change. Small values mean small policy updates.
value_loss: how wrong the value function's predictions are. High and variable means the value function is struggling, which directly hurts learning quality.