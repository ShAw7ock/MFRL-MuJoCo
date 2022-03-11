RL Algos for MUJOCO
============================

# MUJOCO-py
Implementation for Reinforcement Learning algorithms in `MUJOCO` environment. <br>
We use the `PyTorch` as the backend. <br>
Note that we bound the continuous action space into `(-1, 1)` which you can see the details [here](https://github.com/ShAw7ock/mujoco_rl/blob/master/utils/wrappers.py). <br>

# Requirements
* Python >= 3.6.0 (optional)
* PyTorch == 1.7.0 (optional)
* [MUJOCO 200](https://roboti.us/)
* [mujoco-py](https://github.com/openai/mujoco-py)
* OpenAI Gym

# TODO List
- [x] CUDA Supported
- [ ] Off-policy Algos: TD3, DDPG, SAC
- [ ] On-policy Algos: PPO

# Acknowledgement
This code is referenced by [nnaisense/MAGE](https://github.com/nnaisense/MAGE). <br>
