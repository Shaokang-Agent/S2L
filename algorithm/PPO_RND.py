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

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class RND(nn.Module):
    def __init__(self):
        super(RND, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, 192),
            nn.Tanh(),
            nn.Linear(192, 64)
        )
        self.target = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, 192),
            nn.Tanh(),
            nn.Linear(192, 64)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

class PPO_RND():
    def __init__(self, args, agent_id):
        super(PPO_RND, self).__init__()
        self.agent_id = agent_id
        self.action_num = args.action_num
        self.args = args
        self.actor_net = PPO_Actor(self.args)
        self.old_actor_net = PPO_Actor(self.args)
        self.critic_net = PPO_Critic(self.args)
        self.input_net = Shared_input_layer()
        self.rnd = RND()

        if self.args.cuda:
            self.input_net.cuda()
            self.actor_net.cuda()
            self.old_actor_net.cuda()
            self.critic_net.cuda()
            self.rnd.cuda()

        if args.load:
            model_root_path = self.args.save_dir + '/' + 'agent_%d' % self.agent_id + '/'
            if os.path.exists(model_root_path):
                num = max([int(file.split("_")[0]) for file in os.listdir(model_root_path) if file.split("_")[0].isdigit()])
                self.critic_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_critic_net.pkl', map_location='cpu'))
                self.actor_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_actor_net.pkl', map_location='cpu'))
                self.input_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_input_net.pkl', map_location='cpu'))
                self.rnd.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_rnd_net.pkl', map_location='cpu'))
                print('Agent {} successfully loaded PPO_RND-{}'.format(self.agent_id, num))

        self.old_actor_net.load_state_dict(self.actor_net.state_dict())
        self.training_parameters = [{'params': self.actor_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.critic_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.input_net.parameters(), 'lr': self.args.lr}]
        self.training_parameters.append({'params': self.rnd.predictor.parameters(), 'lr': self.args.lr})

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
        batch_next_state = torch.from_numpy(np.array(episode_data['o_next'])[:,agent_id,...]).float()

        self.learn_step_counter+=1
        if self.learn_step_counter % 1000 == 0 and self.args.save:
            self.save_model(num=int(self.learn_step_counter/1000))

        if self.args.cuda:
            batch_observation = batch_observation.cuda()
            batch_action = batch_action.cuda()
            batch_action_prob = batch_action_prob.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        value_loss = 0
        action_loss = 0

        rnd_input = batch_next_state
        rnd_input = rnd_input.permute(0, 3, 1, 2)

        reward = copy.deepcopy(batch_reward)
        reward = reward.unsqueeze(dim=1)
        with torch.no_grad():
            target_next_feature = self.rnd.target(rnd_input)
            predict_next_feature = self.rnd.predictor(rnd_input)
            rnd_reward = (target_next_feature - predict_next_feature).pow(2).mean(dim=1)
            rnd_reward = torch.clamp(rnd_reward, -0.25, 0.25)
            rnd_reward = rnd_reward.unsqueeze(dim=1)
            if self.learn_step_counter % 10 == 0:
                print("env_reward:{}, rnd_reward:{}".format(batch_reward[0].mean(), rnd_reward[0].mean()))
            reward += self.args.shaping_rewards_alpha * rnd_reward

        batch_return = torch.zeros(batch_observation.shape[0], 1)
        if self.args.cuda:
            batch_return = batch_return.cuda()
        for step in range(self.args.num_steps - 1, -1, -1):
            if step == self.args.num_steps - 1:
                batch_return[step, ...] = reward[step, ...]
            if step < self.args.num_steps - 1:
                batch_return[step, ...] = reward[step, ...] + self.args.gamma * batch_return[step + 1, ...]
        #batch_return = (batch_return - batch_return.mean(dim=0).unsqueeze(dim=1)) / (batch_return.std(dim=0).unsqueeze(dim=1) + 1e-7)
        V = self.critic_net(self.input_net(batch_observation))
        advantage = (batch_return - V).detach()
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
            loss = action_loss + value_loss - 0.05 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.args.grad_clip)
            # nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.args.grad_clip)
            # nn.utils.clip_grad_norm_(self.input_net.parameters(), self.args.grad_clip)
            self.optimizer.step()

        predict_next_state_feature, target_next_state_feature = self.rnd(rnd_input)
        forward_loss = torch.mean((predict_next_state_feature-target_next_state_feature.detach()).pow(2), dim=-1)
        if self.args.cuda:
            mask = torch.rand(len(forward_loss)).cuda()
            mask = (mask < 0.2).type(torch.FloatTensor).cuda()
            forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).cuda())
        else:
            mask = torch.rand(len(forward_loss))
            mask = (mask < 0.2).type(torch.FloatTensor)
            forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]))
        self.optimizer.zero_grad()
        forward_loss.backward()
        self.optimizer.step()

        self.old_actor_net.load_state_dict(self.actor_net.state_dict())

        loss_all = {}
        loss_all["actor"] = action_loss
        loss_all["critic"] = value_loss
        loss_all["rnd"] = forward_loss

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
        torch.save(self.rnd.state_dict(), model_path + '/' + str(num) + '_rnd_net.pkl')
