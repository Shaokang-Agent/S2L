from marlgrid.agents import GridAgentInterface
from marlgrid.envs.cluttered import ClutteredMultiGrid
import gym
import numpy as np
from marlgrid.envs import env_from_config
import argparse
from replay_buffer.replay_buffer import ReplayBuffer
from algorithm.DQN import DQN
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
parser.add_argument("--algorithm", type=str, default="DQN")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_episodes", type=int, default=100000)
parser.add_argument("--num_steps", type=int, default=250)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--buffer_size", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=64)
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
args.target_update_iter = 100
args.double_dqn = True
args.grad_norm_clip = 5
args.save_dir = "./runs/" + args.env + "_agent_" + str(args.num_agents) + "_goal_" + str(env_config["n_bonus_tiles"]) + "/" + args.algorithm + "_self_" + str(args.self_learning) + "_rnd_" + str(args.rnd) + "_social_" + str(args.social_learning)
args.log_dir = "./runs/" + args.env + "_agent_" + str(args.num_agents) + "_goal_" + str(env_config["n_bonus_tiles"]) + "/" + args.algorithm + "_self_" + str(args.self_learning) + "_rnd_" + str(args.rnd) + "_social_" + str(args.social_learning) + "/logs"
env_config['agents'] = [player_interface_config for _ in range(args.num_agents)]
env = env_from_config(env_config)
DQN_Agent = [DQN(args, agent_id) for agent_id in range(args.num_agents)]
Repaly_Buffer = ReplayBuffer(args)
writer = SummaryWriter(args.log_dir)
recode_rewards = np.zeros([int(args.num_episodes/args.test_per_epi), args.num_agents])
for epi in tqdm(range(args.num_episodes)):
    args.shaping_rewards_alpha = max([0.05, args.shaping_rewards_alpha-0.0001])
    if args.training:
        for i in range(args.num_agents):
            DQN_Agent[i].init_hidden()
        episode_data = {}
        episode_obs_data = []
        episode_act_data = []
        episode_rew_data = []
        episode_next_obs_data = []
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
            act = []
            for i in range(args.num_agents):
                args.epsilon += 0.00001
                act.append(DQN_Agent[i].choose_action(obs[i], args.epsilon, "training"))
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
            episode_rew_data.append(rew)
            episode_next_obs_data.append(obs_next)
            if args.num_agents > 1 and args.social_learning:
                episode_in_view_data.append(in_view)
            rewards += np.array(rew)
            obs = copy.deepcopy(obs_next)
            if args.render:
                env.render()
        episode_data["o"] = episode_obs_data
        episode_data["u"] = episode_act_data
        episode_data["r"] = episode_rew_data
        episode_data["o_next"] = episode_next_obs_data
        if args.num_agents > 1 and args.social_learning:
            episode_data["in_view"] = episode_in_view_data
        # print("obs: ", np.array(episode_obs_data).shape)
        # print("act: ", np.array(episode_act_data).shape)
        # print("rew: ", np.array(episode_rew_data).shape)
        # print("next_obs: ", np.array(episode_next_obs_data).shape)
        if args.self_learning and args.num_goals > 1:
            if args.num_goals == 2:
                if np.max(np.array(episode_rew_data).sum(axis=0)) >= 2:
                    for _ in range(5):
                        Repaly_Buffer.add(episode_data)
                else:
                    Repaly_Buffer.add(episode_data)
            else:
                if np.max(np.array(episode_rew_data).sum(axis=0)) >= 2 and np.max(np.array(episode_rew_data).sum(axis=0)) < 3:
                    for _ in range(5):
                        Repaly_Buffer.add(episode_data)
                elif np.max(np.array(episode_rew_data).sum(axis=0)) >= 3:
                    for _ in range(10):
                        Repaly_Buffer.add(episode_data)
                else:
                    if np.random.rand() > 0.2:
                        Repaly_Buffer.add(episode_data)
        else:
            Repaly_Buffer.add(episode_data)
        if args.batch_size < Repaly_Buffer.__len__() and epi % 3 == 0:
            mini_batch = Repaly_Buffer.sample(min(Repaly_Buffer.__len__(), args.batch_size))
            for i in range(args.num_agents):
                loss = DQN_Agent[i].learn(mini_batch, i)
                writer.add_scalar("Agent_{}_loss_q".format(str(i)), loss["q_net"], epi)
                if args.rnd:
                    writer.add_scalar("Agent_{}_loss_rnd".format(str(i)), loss["rnd"], epi)
                if args.self_learning:
                    writer.add_scalar("Agent_{}_loss_self_learning".format(str(i)), loss["self_learning"], epi)
                if args.social_learning:
                    writer.add_scalar("Agent_{}_loss_social_learning".format(str(i)), loss["social_learning"], epi)
    if args.test:
        if epi % args.test_per_epi == 0:
            for i in range(args.num_agents):
                DQN_Agent[i].init_hidden()
            obs_dict = env.reset()
            obs = []
            for i in range(args.num_agents):
                obs.append(obs_dict[i]['pov']/255)
            rewards = np.zeros(args.num_agents)
            for step in range(args.num_steps):
                act = []
                for i in range(args.num_agents):
                    act.append(DQN_Agent[i].choose_action(obs[i], 1, "test"))
                obs_next_dict, rew, done, _ = env.step(act)
                obs_next = []
                for i in range(args.num_agents):
                    obs_next.append(obs_next_dict[i]['pov']/255)
                obs = copy.deepcopy(obs_next)
                rewards += np.array(rew)
            print(args.algorithm, "agent:", args.num_agents, "goals:", args.num_goals, epi, rewards)
            recode_rewards[int(epi/args.test_per_epi)] = rewards
            for i in range(args.num_agents):
                writer.add_scalar("Agent_{}_reward".format(str(i)), rewards[i], epi)

np.save(args.save_dir + "/rewards.npy", recode_rewards)