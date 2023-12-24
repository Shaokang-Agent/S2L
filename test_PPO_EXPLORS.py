from marlgrid.agents import GridAgentInterface
from marlgrid.envs.cluttered import ClutteredMultiGrid
import gym
import numpy as np
from marlgrid.envs import env_from_config
import argparse
from replay_buffer.replay_buffer import ReplayBuffer
from algorithm.PPO_EXPLORS import PPO_EXPLORS
from tqdm import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter
import math

parser = argparse.ArgumentParser("Exploration via Self Learning and Social Learning")
# Core training parameters
parser.add_argument("--algorithm", type=str, default="PPO_EXPLORS")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--training_times", type=int, default=3)
parser.add_argument("--num_episodes", type=int, default=100000)
parser.add_argument("--num_steps", type=int, default=250)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--num_agents", type=int, default=3)
parser.add_argument("--num_goals", type=int, default=3)
parser.add_argument("--penalty", type=float, default=-0.1)
parser.add_argument("--action_num", type=int, default=7)
parser.add_argument("--epsilon", type=float, default=0.5)
parser.add_argument("--shaping_rewards_alpha", type=float, default=0.5)
parser.add_argument("--load", type=bool, default=False)
parser.add_argument("--save", type=bool, default=False)
parser.add_argument("--cuda", type=bool, default=False)
parser.add_argument("--render", type=bool, default=False)
parser.add_argument("--training", type=bool, default=True)
parser.add_argument("--test", type=bool, default=True)
parser.add_argument("--test_per_epi", type=int, default=10)

args = parser.parse_args()

env_config = {
    "env_class": "ClutteredGoalCycleEnv",
    "grid_size": 13,
    "max_steps": 250,
    "clutter_density": 0.15,
    "respawn": True,
    "ghost_mode": True,
    "reward_decay": False,
    "n_bonus_tiles": args.num_goals,
    "initial_reward": True,
    "penalty": args.penalty
}

player_interface_config = {
    "view_size": 7,
    "view_offset": 1,
    "view_tile_size": 3,
    "observation_style": "rich",
    "see_through_walls": True,
    "color": "prestige"
}

args.env = env_config["env_class"]
args.state_shape = player_interface_config["view_size"]*player_interface_config["view_tile_size"]
args.grad_clip = 5
args.clip_param = 0.05
args.save_dir = "./runs/" + args.env + "_agent_" + str(args.num_agents) + "_goal_" + str(env_config["n_bonus_tiles"]) + "/" + args.algorithm
args.log_dir = "./runs/" + args.env + "_agent_" + str(args.num_agents) + "_goal_" + str(env_config["n_bonus_tiles"]) + "/" + args.algorithm + "/logs"
# Add the player/agent config to the environment config (as expected by "env_from_config" below)
env_config['agents'] = [player_interface_config for _ in range(args.num_agents)]
env = env_from_config(env_config)

PPO_EXPLORS_Agent = [PPO_EXPLORS(args, agent_id) for agent_id in range(args.num_agents)]
epsilon = 1
writer = SummaryWriter(args.log_dir)
recode_rewards = np.zeros([int(args.num_episodes/args.test_per_epi), args.num_agents])
for epi in tqdm(range(args.num_episodes)):
    args.shaping_rewards_alpha = max([0.05, args.shaping_rewards_alpha - 0.0001])
    if args.training:
        for i in range(args.num_agents):
            PPO_EXPLORS_Agent[i].init_hidden()
        episode_obs_data, episode_act_data, epi_aprobability_data, episode_rew_data, episode_next_obs_data = [], [], [], [], []
        obs_dict = env.reset()
        obs = []
        for i in range(args.num_agents):
            obs.append(obs_dict[i]['pov']/255)
        rewards = np.zeros(args.num_agents)
        for step in range(args.num_steps):
            act, act_probability = [],[]
            for i in range(args.num_agents):
                args.epsilon += 0.00001
                action, action_logprobability = PPO_EXPLORS_Agent[i].choose_action(obs[i], args.epsilon)
                act.append(action)
                act_probability.append(action_logprobability)
            obs_next_dict, rew, done, _ = env.step(act)
            obs_next = []
            for i in range(args.num_agents):
                obs_next.append(obs_next_dict[i]['pov']/255)
            episode_obs_data.append(obs)
            episode_act_data.append(act)
            epi_aprobability_data.append(act_probability)
            episode_rew_data.append(rew)
            episode_next_obs_data.append(obs_next)
            rewards += np.array(rew)
            obs = copy.deepcopy(obs_next)
            if args.render:
                env.render()
            episode = dict(o=episode_obs_data.copy(),
                           u=episode_act_data.copy(),
                           u_probability=epi_aprobability_data.copy(),
                           r=episode_rew_data.copy()
                           )

        for i in range(args.num_agents):
            loss = PPO_EXPLORS_Agent[i].learn(episode, i)
            writer.add_scalar("Agent_{}_loss_actor".format(str(i)), loss["actor"], epi)
            writer.add_scalar("Agent_{}_loss_critic".format(str(i)), loss["critic"], epi)
            writer.add_scalar("Agent_{}_loss_explors_reward".format(str(i)), loss["loss_explors_reward"], epi)
            writer.add_scalar("Agent_{}_loss_explors_critic".format(str(i)), loss["loss_explors_critic"], epi)

    if args.test:
        if epi % args.test_per_epi == 0:
            for i in range(args.num_agents):
                PPO_EXPLORS_Agent[i].init_hidden()
            obs_dict = env.reset()
            obs = []
            for i in range(args.num_agents):
                obs.append(obs_dict[i]['pov']/255)
            rewards = np.zeros(args.num_agents)
            for step in range(args.num_steps):
                act = []
                for i in range(args.num_agents):
                    action, _ = PPO_EXPLORS_Agent[i].choose_action(obs[i], 1)
                    act.append(action)
                obs_next_dict, rew, done, _ = env.step(act)
                obs_next = []
                for i in range(args.num_agents):
                    obs_next.append(obs_next_dict[i]['pov']/255)
                rewards += np.array(rew)
                obs = copy.deepcopy(obs_next)
            print(args.algorithm, "agent:", args.num_agents, "goals:", args.num_goals, epi, rewards)
            recode_rewards[int(epi / args.test_per_epi)] = rewards
            for i in range(args.num_agents):
                writer.add_scalar("Agent_{}_reward".format(str(i)), rewards[i], epi)

np.save(args.save_dir + "/rewards.npy", recode_rewards)

