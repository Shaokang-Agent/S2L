import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
from torch.distributions import Categorical
from torch.nn.parameter import Parameter

class My_Linear(nn.Module):
    def __init__(self, args_cuda, in_features, out_features, bias=True):
        super(My_Linear, self).__init__()
        self.args_cuda = args_cuda
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = Parameter(torch.Tensor(out_features, in_features))
        self.weight_mean.data.uniform_(-0.1, 0.1)
        self.weight_std = Parameter(torch.Tensor(out_features, in_features))
        self.weight_std.data.uniform_(-0.1, 0.1)
        if bias:
            self.bias_mean = Parameter(torch.Tensor(out_features))
            self.bias_mean.data.uniform_(-0.1, 0.1)
            self.bias_std = Parameter(torch.Tensor(out_features))
            self.bias_std.data.uniform_(-0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        a = torch.rand([self.out_features, self.in_features])
        b = torch.rand([self.out_features])
        if self.args_cuda:
            a = a.cuda()
            b = b.cuda()
        self.weight = self.weight_mean + self.weight_std.mul(a)
        self.bias = self.bias_mean + self.bias_std.mul(b)
        return F.linear(input, self.weight, self.bias)

class Shared_input_layer(nn.Module):
    def __init__(self):
        super(Shared_input_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(576, 192)
        self.lstm = nn.LSTM(input_size=192, hidden_size=192, num_layers=1, batch_first=True)

    def forward(self, x, h0=None, c0=None):
        batch, step, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        x = x.reshape(batch * step, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.reshape(batch, step, -1)
        x = F.tanh(self.fc(x))
        if h0 == None:
            x, (h, c) = self.lstm(x)
            return x
        else:
            x, (h, c) = self.lstm(x, (h0, c0))
            return x, h, c

class QNet(nn.Module):
    """docstring for Net"""
    def __init__(self, args):
        super(QNet, self).__init__()
        self.args = args
        self.qvalue1 = My_Linear(args.cuda, 192, 64)
        self.qvalue2 = My_Linear(args.cuda, 64, 64)
        self.qvalue3 = My_Linear(args.cuda, 64, 7)
        
    def forward(self, x):
        x = F.tanh(self.qvalue1(x))
        x = F.tanh(self.qvalue2(x))
        x = self.qvalue3(x)
        return x

class DQN_NOISYNET():
    """docstring for DQN"""
    def __init__(self, args, agent_id):
        super(DQN_NOISYNET, self).__init__()
        self.agent_id = agent_id
        self.args = args
        self.input_net = Shared_input_layer()
        self.eval_net, self.target_net = QNet(args).float(), QNet(args).float()
        if args.cuda:
            self.input_net.cuda()
            self.eval_net.cuda()
            self.target_net.cuda()
        if args.load:
            model_root_path = self.args.save_dir + '/' + 'agent_%d' % self.agent_id + '/'
            if os.path.exists(model_root_path):
                num = max([int(file.split("_")[0]) for file in os.listdir(model_root_path) if file.split("_")[0].isdigit()])
                self.eval_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_Q_Net.pkl', map_location='cpu'))
                self.input_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_input_net.pkl', map_location='cpu'))
                print('Agent {} successfully loaded DQN_NOISYNET-{}'.format(self.agent_id, num))

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.action_num = self.args.action_num
        self.learn_step_counter = 0
        self.training_parameters = [{'params': self.eval_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.input_net.parameters(), 'lr': self.args.lr}]

        self.optimizer = torch.optim.Adam(self.training_parameters)
        self.loss_function = torch.nn.MSELoss()

    def init_hidden(self):
        self.h0, self.c0 = torch.randn(1, 1, 192),torch.randn(1, 1, 192)

    def choose_action(self, obs, episolon, flag):
        if flag == "training":
            if np.random.rand() <= episolon:
                obsnp = np.array(obs)
                obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0), 0)
                obs = obs.float()
                if self.args.cuda:
                    obs = obs.cuda()
                with torch.no_grad():
                    if self.args.cuda:
                        self.h0 = self.h0.cuda()
                        self.c0 = self.c0.cuda()
                    x, self.h0, self.c0 = self.input_net(obs, self.h0, self.c0)
                    action_value = self.eval_net(x)
                    action_value = action_value.squeeze(dim=0).squeeze(dim=0)
                    action_prob = F.softmax(action_value / 0.1, dim=-1)
                    dist = Categorical(action_prob)
                    action = dist.sample()
                    action = action.cpu().data.numpy()
            else:
                action = np.random.randint(0, self.action_num)
            return action
        else:
            obsnp = np.array(obs)
            obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0), 0)
            obs = obs.float()
            if self.args.cuda:
                obs = obs.cuda()
            with torch.no_grad():
                if self.args.cuda:
                    self.h0 = self.h0.cuda()
                    self.c0 = self.c0.cuda()
                x, self.h0, self.c0 = self.input_net(obs, self.h0, self.c0)
                action_value = self.eval_net(x)
                action_value = action_value.squeeze(dim=0)
                action = torch.max(action_value, 1)[1].cpu().data.numpy()
                action = action[0]
                return action

    def learn(self, episode_data, agent_id):
        if self.learn_step_counter % self.args.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter+=1

        if self.learn_step_counter % 1000 == 0 and self.args.save:
            self.save_model(num=int(self.learn_step_counter/1000))

        batch_state = torch.from_numpy(episode_data['o'][:,:,agent_id,...]).float()
        batch_action = torch.from_numpy(episode_data['u'][:,:,agent_id,...]).long()
        batch_reward = torch.from_numpy(episode_data['r'][:,:, agent_id, ...]).float()
        batch_next_state = torch.from_numpy(episode_data['o_next'][:,:,agent_id,...]).float()
        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        #print(batch_action.shape)
        q_eval = self.eval_net(self.input_net(batch_state)).gather(2, batch_action)
        q_next = self.target_net(self.input_net(batch_next_state)).detach()
        if self.learn_step_counter % 10 == 0:
            print(self.eval_net(self.input_net(batch_state))[0], self.target_net(self.input_net(batch_next_state))[0])
        if self.args.double_dqn:
            q_target = batch_reward + self.args.gamma * q_next.gather(2, self.eval_net(self.input_net(batch_next_state)).max(2)[1].unsqueeze(dim=2))
        else:
            q_target = batch_reward + self.args.gamma * q_next.max(2)[0].unsqueeze(dim=2)
        q_target = q_target.detach()

        loss = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()
        return loss

    def save_model(self,num):
        model_path = self.args.save_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.eval_net.state_dict(), model_path + '/' + str(num) + '_Q_Net.pkl')
        torch.save(self.input_net.state_dict(), model_path + '/' + str(num) + '_input_net.pkl')
