Tools used:
1. dnSpy

TO DO:
1. Generate more game play
2. Test sector assignment
3. Test entire ray implementation
4. write ImitationLearning class. should contain method for transformating recording.txt into state data, and do Imitationlearning stuff

# Training term definitions

### rollout section — describes the experience collected this iteration:

**ep_len_mean**: average number of steps per episode.

**ep_rew_mean**: average total reward per episode. The most important metric for tracking whether your agent is improving.

**time section** — bookkeeping:

**fps**: steps per second. Ours is 0-1 because each Jump King step takes several real seconds.

**iterations**: how many n_steps buffers have been collected so far.

**time_elapsed**: seconds since training started.

**total_timesteps**: total steps taken this session.

**train section** — describes the neural network update:

**approx_kl**: how much the policy changed during this update. Too high means unstable updates, too low means barely learning. Ideally stays small but nonzero.

**clip_fraction**: fraction of updates that hit PPO's clipping threshold. Near 0 means the policy is barely changing — could indicate the learning rate is too low or the policy has stagnated.

**clip_range**: the clipping threshold itself, fixed at 0.2 by default.

**entropy_loss**: measures how random the policy is. High magnitude means more exploratory, decreasing over time means the policy is becoming more confident.

**explained_variance**: how well the value function predicts actual returns. Near 0 means it's essentially guessing. Near 1 means it's accurately predicting future rewards — this is what you want.

**learning_rate**: your current learning rate, fixed at 0.0003.

**loss**: combined loss of the policy and value function. Variable and hard to interpret in isolation.

**n_updates**: cumulative number of gradient updates applied to the network.

**policy_gradient_loss**: how much the policy is being pushed to change. Small values mean small policy updates.

**value_loss**: how wrong the value function's predictions are. High and variable means the value function is struggling, which directly hurts learning quality.

# Strategies used
* State space design decisions. Started with (x, y, x_vel, y_vel), then added current screen. If JK is being played optimally, we'll never see the same set of coordinates, so agent can never extrapolate what it's learned from optimal play under one state to optimal play under another state. Therefore, we need to add platform information. New state space is now: (x, y, x_vel, y_vel, dist_to_left_wall, dist_to_right_wall, current_screen, closest_platform_up_left, closest_platform_up_right, closest_platform_left, closest_platform_right, closest_platform_up_left_next_screen, closest_platform_up_right_next_screen). Each closest platform is a tuple of coordinates, flattened when passed to the agent. Platform information is taken from a custom C# mod that writes game data and platform info to a file.