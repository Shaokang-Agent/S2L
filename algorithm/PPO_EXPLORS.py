import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import os
import copy

class Shared_input_layer(nn.Module):
    def __init__(self):
        super(Shared_input_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(576, 192)
        self.lstm = nn.LSTM(input_size=192, hidden_size=192, num_layers=1, batch_first=True)

    def forward(self, x, h0=None, c0=None):
        step, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0, 3, 1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.reshape(step, -1)
        x = F.tanh(self.fc(x))
        if h0 == None:
            x = torch.unsqueeze(x, dim=0)
            x, (h, c) = self.lstm(x)
            x = torch.squeeze(x, dim=0)
            return x
        else:
            x = torch.unsqueeze(x, dim=0)
            x, (h, c) = self.lstm(x, (h0, c0))
            x = torch.squeeze(x, dim=0)
            return x, h, c

class PPO_Actor(nn.Module):
    def __init__(self, args):
        super(PPO_Actor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(192, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.gumbel_softmax(self.fc3(x), dim=-1)
        return x

class PPO_Critic(nn.Module):
    def __init__(self, args):
        super(PPO_Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(192, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class Self_reward_Net(nn.Module):
    """docstring for Net"""
    def __init__(self, args):
        super(Self_reward_Net, self).__init__()
        self.args = args
        self.selfvalue1 = nn.Linear(192, 64)
        self.selfvalue2 = nn.Linear(64, 64)
        self.selfvalue3 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.tanh(self.selfvalue1(x))
        x = F.tanh(self.selfvalue2(x))
        x = F.tanh(self.selfvalue3(x))
        return x

class Self_critic_Net(nn.Module):
    """docstring for Net"""
    def __init__(self, args):
        super(Self_critic_Net, self).__init__()
        self.args = args
        self.selfvalue1 = nn.Linear(192, 64)
        self.selfvalue2 = nn.Linear(64, 64)
        self.selfvalue3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.selfvalue1(x))
        x = F.tanh(self.selfvalue2(x))
        x = self.selfvalue3(x)
        return x

class PPO_EXPLORS():
    def __init__(self, args, agent_id):
        super(PPO_EXPLORS, self).__init__()
        self.agent_id = agent_id
        self.action_num = args.action_num
        self.args = args
        self.actor_net = PPO_Actor(self.args)
        self.old_actor_net = PPO_Actor(self.args)
        self.critic_net = PPO_Critic(self.args)
        self.input_net = Shared_input_layer()
        self.explore_reward_net = Self_reward_Net(self.args)
        self.explore_critic = Self_critic_Net(self.args)
        if self.args.cuda:
            self.input_net.cuda()
            self.actor_net.cuda()
            self.old_actor_net.cuda()
            self.critic_net.cuda()
            self.explore_reward_net.cuda()
            self.explore_critic.cuda()
        if args.load:
            model_root_path = self.args.save_dir + '/' + 'agent_%d' % self.agent_id + '/'
            if os.path.exists(model_root_path):
                num = max([int(file.split("_")[0]) for file in os.listdir(model_root_path) if file.split("_")[0].isdigit()])
                self.critic_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_critic_net.pkl', map_location='cpu'))
                self.actor_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_actor_net.pkl', map_location='cpu'))
                self.input_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_input_net.pkl', map_location='cpu'))
                self.explore_reward_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_explore_net.pkl', map_location='cpu'))
                self.explore_critic.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_explore_critic_net.pkl', map_location='cpu'))
                print('Agent {} successfully loaded PPO_EXPLORS-{}'.format(self.agent_id, num))

        self.old_actor_net.load_state_dict(self.actor_net.state_dict())
        self.training_parameters = [{'params': self.actor_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.critic_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.input_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.explore_reward_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.explore_critic.parameters(), 'lr': self.args.lr}]

        self.optimizer = torch.optim.Adam(self.training_parameters)
        self.learn_step_counter = 0
        self.loss_function = torch.nn.MSELoss()

    def init_hidden(self):
        self.h0, self.c0 = torch.randn(1, 1, 192),torch.randn(1, 1, 192)

    def choose_action(self, obs, episolon):
        obsnp = np.array(obs)
        obs = torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0)
        obs = obs.float()
        if self.args.cuda:
            obs = obs.cuda()
        with torch.no_grad():
            if self.args.cuda:
                self.h0 = self.h0.cuda()
                self.c0 = self.c0.cuda()
            x, self.h0, self.c0 = self.input_net(obs, self.h0, self.c0)
            action_prob = self.old_actor_net(x)
            dist = Categorical(action_prob)
            if np.random.rand() <= episolon:
                action = dist.sample()
                action_logprob = dist.log_prob(action)
            else:
                action = torch.randint(0, self.action_num, (1,))
                if self.args.cuda:
                    action = action.cuda()
                action_logprob = dist.log_prob(action)
            action = action.squeeze().cpu().data.numpy()
            action_logprob = action_logprob.squeeze().squeeze().cpu().data.numpy()
        return int(action), action_logprob

    def learn(self, episode_data, agent_id):
        batch_observation = torch.from_numpy(np.array(episode_data['o'])[:,agent_id,...]).float()
        batch_action = torch.from_numpy(np.array(episode_data['u'])[:,agent_id,...]).long()
        batch_action_prob = torch.from_numpy(np.array(episode_data['u_probability'])[:,agent_id,...]).float()
        batch_reward = torch.from_numpy(np.array(episode_data['r'])[:,agent_id,...]).float()

        self.learn_step_counter+=1
        if self.learn_step_counter % 1000 == 0 and self.args.save:
            self.save_model(num=int(self.learn_step_counter/1000))

        if self.args.cuda:
            batch_observation = batch_observation.cuda()
            batch_action = batch_action.cuda()
            batch_action_prob = batch_action_prob.cuda()
            batch_reward = batch_reward.cuda()

        shaping_reward = self.explore_reward_net(self.input_net(batch_observation)).gather(1, batch_action.unsqueeze(dim=1)).detach()
        print("Env reaward: {}, ExploRS reward {}".format(batch_reward.mean(), shaping_reward.mean()))
        batch_return = torch.zeros(batch_observation.shape[0], 1)
        if self.args.cuda:
            batch_return = batch_return.cuda()
        for step in range(self.args.num_steps - 1, -1, -1):
            if step == self.args.num_steps - 1:
                batch_return[step, ...] = batch_reward[step, ...] + self.args.shaping_rewards_alpha * shaping_reward[step, ...]
            if step < self.args.num_steps - 1:
                batch_return[step, ...] = batch_reward[step, ...] + self.args.shaping_rewards_alpha * shaping_reward[step, ...] + self.args.gamma * batch_return[step + 1, ...]
        #batch_return = (batch_return - batch_return.mean(dim=0).unsqueeze(dim=1)) / (batch_return.std(dim=0).unsqueeze(dim=1) + 1e-7)
        V = self.critic_net(self.input_net(batch_observation))
        advantage = (batch_return - V).detach()
        action_loss = 0
        value_loss = 0
        for _ in range(self.args.training_times):
            # epoch iteration, PPO core!!!
            action_probs = self.actor_net(self.input_net(batch_observation)) # new policy
            if self.learn_step_counter % 10 == 0:
                print(action_probs[0])
            dist = Categorical(action_probs)
            action_prob = dist.log_prob(batch_action).squeeze()
            entropy = dist.entropy()
            ratio = torch.exp(action_prob - batch_action_prob).unsqueeze(dim=1)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * advantage
            action_loss = -torch.min(surr1, surr2).mean()
            V = self.critic_net(self.input_net(batch_observation))
            value_loss = F.mse_loss(batch_return, V)
            loss = action_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.args.grad_clip)
            # nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.args.grad_clip)
            # nn.utils.clip_grad_norm_(self.input_net.parameters(), self.args.grad_clip)
            self.optimizer.step()

        batch_intial_return = torch.zeros_like(batch_reward)
        if self.args.cuda:
            batch_intial_return = batch_intial_return.cuda()
        for step in range(self.args.num_steps - 1, -1, -1):
            if step == self.args.num_steps - 1:
                batch_intial_return[step, ...] = batch_reward[step, ...]
            if step < self.args.num_steps - 1:
                batch_intial_return[step, ...] = batch_reward[step, ...] + self.args.gamma * batch_intial_return[step + 1,...]
        explore_critic_eval = self.explore_critic(self.input_net(batch_observation)).detach()
        explore_reward_sa = self.explore_reward_net(self.input_net(batch_observation)).gather(1, batch_action.unsqueeze(dim=1))
        explore_reward_sa_other_pi = self.explore_reward_net(self.input_net(batch_observation))
        sa_other_pi = self.old_actor_net(self.input_net(batch_observation)).detach()
        explore_reward_sa_other = torch.sum(explore_reward_sa_other_pi * sa_other_pi, dim=-1).unsqueeze(dim=1)
        loss_explore = -torch.mean((batch_intial_return - explore_critic_eval) * (explore_reward_sa - explore_reward_sa_other))
        self.optimizer.zero_grad()
        loss_explore.backward()
        # torch.nn.utils.clip_grad_norm_(self.input_net.parameters(), self.args.grad_norm_clip)
        # torch.nn.utils.clip_grad_norm_(self.explore_reward_net.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()

        explore_critic = self.explore_critic(self.input_net(batch_observation))
        loss_critic = nn.functional.smooth_l1_loss(batch_intial_return, explore_critic)
        self.optimizer.zero_grad()
        loss_critic.backward()
        # torch.nn.utils.clip_grad_norm_(self.explore_critic.parameters(), self.args.grad_norm_clip)
        # torch.nn.utils.clip_grad_norm_(self.input_net.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()

        self.old_actor_net.load_state_dict(self.actor_net.state_dict())
        loss_all = {}
        loss_all["actor"] = action_loss
        loss_all["critic"] = value_loss
        loss_all["loss_explors_reward"] = loss_explore
        loss_all["loss_explors_critic"] = loss_critic
        return loss_all

    def save_model(self,num):
        model_path = self.args.save_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.critic_net.state_dict(), model_path + '/' + str(num) + '_critic_net.pkl')
        torch.save(self.actor_net.state_dict(), model_path + '/' + str(num) + '_actor_net.pkl')
        torch.save(self.input_net.state_dict(), model_path + '/' + str(num) + '_input_net.pkl')
        torch.save(self.explore_reward_net.state_dict(), model_path + '/' + str(num) + '_explore_net.pkl')
        torch.save(self.explore_critic.state_dict(), model_path + '/' + str(num) + '_explore_critic_net.pkl')
