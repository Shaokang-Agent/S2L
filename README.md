# Implementation of the paper "Multi-Agent Exploration via Self-Learning and Social Learning"


## Goal Cycle Environment
Agents receive a reward of $+1$ for successfully traversing between multiple goals in the correct order, and incur a penalty of $-0.1$ for deviating from this order. In all other instances, the reward remains at $0$. We have removed experts in the Goal Cycle environment, rendering all agents decentralized learners. Therefore, the sparse reward and the intricate sequential structure present a formidable challenge for multi-agent exploration.


## Installation
```
# Create a python virtual environment with python=3.7.16
conda create -n S2L python=3.7.16
# Activate the environment
conda activate S2L
# Install the requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Training


```
# For example: Training DQN_S2L algorithm
python test_DQN_S2L.py

# Training PPO_S2L algorithm:
python test_PPO_S2L.py

# Training PPO algorithm:
python test_PPO.py 
...
```

## Results
The rewards of all agents will be stored in the file: /runs.

## Cite our paper
```
@inproceedings{S2L,
  author       = {Shaokang Dong and
                  Chao Li and
                  Wubing Chen and
                  Hongye Cao and
                  Wenbin Li and
                  Yang Gao},
  title        = {Multi-Agent Exploration via Self-Learning and Social Learning},
  booktitle    = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
                  {ICASSP} 2024, Seoul, Republic of Korea, April 14-19, 2024},
  pages        = {5055--5059},
  publisher    = {{IEEE}},
  year         = {2024},
  doi          = {10.1109/ICASSP48485.2024.10446068},
  timestamp    = {Mon, 05 Aug 2024 15:27:25 +0200}
}
```
