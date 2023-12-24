from marlgrid.agents import GridAgentInterface
from marlgrid.envs.cluttered import ClutteredMultiGrid
import gym
import numpy as np
from marlgrid.envs import env_from_config
import argparse
from replay_buffer.replay_buffer import ReplayBuffer
from algorithm.PPO import PPO
from tqdm import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter
import math
def others_in_view(self_pos, self_dir, others_pos):
    if self_dir == 0:
        if (others_pos[0]-self_pos[0] >= -1 or others_pos[0]-self_pos[0] <= 5) and (others_pos[1]-self_pos[1] >= -3 or others_pos[1]-self_pos[1] <= 3):
            return True
        else:
            return False
    elif self_dir == 1:
        if (others_pos[1]-self_pos[1] >= -1 or others_pos[1]-self_pos[0] <= 5) and (others_pos[0]-self_pos[0] >= -3 or others_pos[0]-self_pos[0] <= 3):
            return True
        else:
            return False
    elif self_dir == 2:
        if (others_pos[0]-self_pos[0] >= -5 or others_pos[0]-self_pos[0] <= 1) and (others_pos[1]-self_pos[1] >= -3 or others_pos[1]-self_pos[1] <= 3):
            return True
        else:
            return False
    else:
        if (others_pos[1]-self_pos[1] >= -5 or others_pos[1]-self_pos[0] <= 1) and (others_pos[0]-self_pos[0] >= -3 or others_pos[0]-self_pos[0] <= 3):
            return True
        else:
            return False

parser = argparse.ArgumentParser("Exploration via Self Learning and Social Learning")
# Core training parameters
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--training_times", type=int, default=5)
parser.add_argument("--num_episodes", type=int, default=100000)
parser.add_argument("--num_steps", type=int, default=250)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--num_agents", type=int, default=3)
parser.add_argument("--num_goals", type=int, default=3)
parser.add_argument("--penalty", type=float, default=-0.1)
parser.add_argument("--action_num", type=int, default=7)
parser.add_argument("--epsilon", type=float, default=0.5)
parser.add_argument("--load", type=bool, default=False)
parser.add_argument("--save", type=bool, default=False)
parser.add_argument("--rnd", type=bool, default=False)
parser.add_argument("--self_learning", type=bool, default=False)
parser.add_argument("--shaping_rewards_alpha", type=float, default=0.5)
parser.add_argument("--social_learning", type=bool, default=False)
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
args.save_dir = "./runs/" + args.env + "_agent_" + str(args.num_agents) + "_goal_" + str(env_config["n_bonus_tiles"]) + "/" + args.algorithm + "_self_" + str(args.self_learning) + "_rnd_" + str(args.rnd) + "_social_" + str(args.social_learning)
args.log_dir = "./runs/" + args.env + "_agent_" + str(args.num_agents) + "_goal_" + str(env_config["n_bonus_tiles"]) + "/" + args.algorithm + "_self_" + str(args.self_learning) + "_rnd_" + str(args.rnd) + "_social_" + str(args.social_learning) + "/logs"
# Add the player/agent config to the environment config (as expected by "env_from_config" below)
env_config['agents'] = [player_interface_config for _ in range(args.num_agents)]
env = env_from_config(env_config)

PPO_Agent = [PPO(args, agent_id) for agent_id in range(args.num_agents)]
epsilon = 1
writer = SummaryWriter(args.log_dir)
recode_rewards = np.zeros([int(args.num_episodes/args.test_per_epi), args.num_agents])
for epi in tqdm(range(args.num_episodes)):
    args.shaping_rewards_alpha = max([0.05, args.shaping_rewards_alpha - 0.0001])
    if args.training:
        for i in range(args.num_agents):
            PPO_Agent[i].init_hidden()
        episode_obs_data, episode_act_data, epi_aprobability_data, episode_rew_data, episode_next_obs_data = [], [], [], [], []
        if args.num_agents > 1 and args.social_learning:
            episode_in_view_data = []
        obs_dict = env.reset()
        obs = []
        for i in range(args.num_agents):
            obs.append(obs_dict[i]['pov']/255)
        rewards = np.zeros(args.num_agents)
        for step in range(args.num_steps):
            if args.num_agents > 1 and args.social_learning:
                pos = []
                dir = []
                in_view = np.zeros([args.num_agents, args.num_agents - 1])
            act, act_probability = [],[]
            for i in range(args.num_agents):
                args.epsilon += 0.00001
                action, action_logprobability = PPO_Agent[i].choose_action(obs[i], args.epsilon)
                act.append(action)
                act_probability.append(action_logprobability)
            obs_next_dict, rew, done, _ = env.step(act)
            obs_next = []
            for i in range(args.num_agents):
                if args.num_agents > 1 and args.social_learning:
                    pos.append(env.agents[i].pos)
                    dir.append(env.agents[i].dir)
                obs_next.append(obs_next_dict[i]['pov']/255)
            if args.num_agents > 1 and args.social_learning:
                for i in range(args.num_agents):
                    j = 0
                    for others in range(args.num_agents):
                        if others != i:
                            if others_in_view(pos[i], dir[i], pos[others]):
                                in_view[i, j] = 1
                            else:
                                in_view[i, j] = 0
                            j += 1
            episode_obs_data.append(obs)
            episode_act_data.append(act)
            epi_aprobability_data.append(act_probability)
            episode_rew_data.append(rew)
            episode_next_obs_data.append(obs_next)
            if args.num_agents > 1 and args.social_learning:
                episode_in_view_data.append(in_view)
            rewards += np.array(rew)
            obs = copy.deepcopy(obs_next)
            if args.render:
                env.render()
        if args.num_agents > 1 and args.social_learning:
            episode = dict(o=episode_obs_data.copy(),
                           u=episode_act_data.copy(),
                           u_probability=epi_aprobability_data.copy(),
                           r=episode_rew_data.copy(),
                           o_next=episode_next_obs_data.copy(),
                           in_view=episode_in_view_data.copy()
                           )
        else:
            episode = dict(o=episode_obs_data.copy(),
                           u=episode_act_data.copy(),
                           u_probability=epi_aprobability_data.copy(),
                           r=episode_rew_data.copy(),
                           o_next=episode_next_obs_data.copy()
                           )

        for i in range(args.num_agents):
            no_training_flag = 0
            # if args.self_learning and args.num_goals > 1:
            #     if args.num_goals == 2:
            #         if np.max(np.array(episode_rew_data).sum(axis=0)) >= 2:
            #             for _ in range(2):
            #                 loss = PPO_Agent[i].learn(episode, i)
            #         else:
            #             loss = PPO_Agent[i].learn(episode, i)
            #     else:
            #         if np.max(np.array(episode_rew_data).sum(axis=0)) >= 2 and np.max(
            #                 np.array(episode_rew_data).sum(axis=0)) < 3:
            #             for _ in range(2):
            #                 loss = PPO_Agent[i].learn(episode, i)
            #         elif np.max(np.array(episode_rew_data).sum(axis=0)) >= 3:
            #             for _ in range(3):
            #                 loss = PPO_Agent[i].learn(episode, i)
            #         else:
            #             if np.random.rand() > 0.2:
            #                 loss = PPO_Agent[i].learn(episode, i)
            #             else:
            #                 no_training_flag = 1
            # else:
            #     loss = PPO_Agent[i].learn(episode, i)
            loss = PPO_Agent[i].learn(episode, i)
            if no_training_flag == 0:
                writer.add_scalar("Agent_{}_loss_actor".format(str(i)), loss["actor"], epi)
                writer.add_scalar("Agent_{}_loss_critic".format(str(i)), loss["critic"], epi)
                if args.rnd:
                    writer.add_scalar("Agent_{}_loss_rnd".format(str(i)), loss["rnd"], epi)
                if args.self_learning:
                    writer.add_scalar("Agent_{}_loss_self_learning".format(str(i)), loss["self_learning"], epi)
                if args.social_learning:
                    writer.add_scalar("Agent_{}_loss_social_learning".format(str(i)), loss["social_learning"], epi)
    if args.test:
        if epi % args.test_per_epi == 0:
            for i in range(args.num_agents):
                PPO_Agent[i].init_hidden()
            obs_dict = env.reset()
            obs = []
            for i in range(args.num_agents):
                obs.append(obs_dict[i]['pov']/255)
            rewards = np.zeros(args.num_agents)
            for step in range(args.num_steps):
                act = []
                for i in range(args.num_agents):
                    action, _ = PPO_Agent[i].choose_action(obs[i], 1)
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