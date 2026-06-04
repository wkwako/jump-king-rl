# Description

I created a reinforcement learning (RL) agent to play Jump King, a difficult platformer that requires precise jumps, practice, and persistence to beat. The agent is provided game data in real-time that it uses to select actions and execute them, and learns to maximize reward signals that it receives from completing these actions.

Jump King is deterministic in theory, but its physics engine can't be perfectly simulated from outside the game without a full recreation, making planning intractable. It's a game that looks like it should be plannable, but isn't. This makes it an interesting testbed for comparing learned approaches like RL, BC, and hybrid methods.

*Trained BC+RL agent playing Jump King, screens 8-9*
![Agent playing Jump King](agent_demo.gif)

# Results

The agent has completed 11 of the 43 screens in the base game. Most screens can be completed 95% of the time without falls. Chained together, these 11 screens are reliably completed in ~3 minutes. Behavioral cloning (BC) initialization from user-recorded inputs significantly reduces training time versus pure RL. Full comparative results (BC, RL, BC+RL, hardcoded baseline) are in progress and will be documented in the full technical writeup.

# Architecture

* **C# game mod**: extracts real-time game data (player position, velocity, screen, etc.) and sends it to Python via a named pipe

* **Python Environment**: Utilizes a custom Gymnasium framework to receive game data, generate state information for PPO, execute actions, and compute rewards

* **BC Pretraining**: Policy network initialized using ~10 human playthroughs before RL training begins

* **PPO agent (stable-baselines3)**: Fine-tunes the BC policy via reinforcement learning, optimizing a reward function primarily based on height gain, proximity to goal, and screen completion 

* **Model manager**: Manages creation, loading, saving, and overwriting of models and metadata, in addition to switching, logging, and per-screen training

# Links:

* **Technical writeup**: *in progress*