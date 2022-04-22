Model-Free Reinforcment Learning (MFRL)
============================

# NOTE
* Implementation for MFRL in `MuJoCo` environment using `PyTorch` as backend.
* Run the experiments as:

`python main.py --env_name ENV_NAME --algo ALGO_NAME --use_cuda True`

* We bound the continuous action space into `[-1, 1]` which can be seen in: `./utils/wrappers.py` and the supported `ENV_NAME` can be seen in: `./envs/gymmb/__init__.py`.
* Supported RL algorithms `ALGO_NAME` can be seen in: `./algos/__init__.py`
* Modify the Hyper-parameters in: `./components/arguments.py`

# Requirements
* Python >= 3.6.0 (optional)
* PyTorch == 1.7.0 (optional)
* [MUJOCO 200](https://roboti.us/)
* [mujoco-py](https://github.com/openai/mujoco-py)
* OpenAI Gym

# TODO List
- [x] CUDA Supported
- [x] Off-policy Algos: TD3, DDPG, SAC
- [ ] On-policy Algos: PPO

# Acknowledgement
This code is referenced by [nnaisense/MAGE](https://github.com/nnaisense/MAGE).
