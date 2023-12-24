import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
from torch.distributions import Categorical

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
        self.qvalue1 = nn.Linear(192, 64)
        self.qvalue2 = nn.Linear(64, 64)
        self.qvalue3 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.tanh(self.qvalue1(x))
        x = F.tanh(self.qvalue2(x))
        x = self.qvalue3(x)
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
        #x = self.selfvalue3(x)
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

class Social_learning_net(nn.Module):
    def __init__(self, args):
        super(Social_learning_net, self).__init__()
        self.args = args
        self.fc = nn.Linear(192+7, 576)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=3,stride=3)

        self.others_policy_fc = nn.Linear(192+7, 64)
        self.others_policy_out = nn.Linear(64, 7*(self.args.num_agents-1))

    def forward(self, x, a):
        batch, step = x.shape[0], x.shape[1]
        onehot_a = F.one_hot(a, num_classes=7)
        onehot_a = onehot_a.squeeze(dim=2)
        input = torch.cat([x, onehot_a], dim=2)
        x = F.tanh(self.fc(input))
        x = x.reshape(batch*step, 64, 3, 3)
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = F.leaky_relu(self.deconv3(x))
        x = x.reshape(batch, step, 3, 21, 21)
        x = x.permute(0, 1, 3, 4, 2)

        y = F.tanh(self.others_policy_fc(input))
        y = self.others_policy_out(y)
        y = y.reshape(batch, step, self.args.num_agents-1, 7)
        return x,y

class DQN():
    """docstring for DQN"""
    def __init__(self, args, agent_id):
        super(DQN, self).__init__()
        self.agent_id = agent_id
        self.args = args
        self.input_net = Shared_input_layer()
        self.eval_net, self.target_net = QNet(args).float(), QNet(args).float()
        if args.self_learning:
            self.explore_reward_net = Self_reward_Net(self.args)
        if self.args.social_learning:
            self.social_learning_net = Social_learning_net(self.args)
        if args.rnd:
            self.rnd = RND()
        if args.cuda:
            self.input_net.cuda()
            self.eval_net.cuda()
            self.target_net.cuda()
            if args.self_learning:
                self.explore_reward_net.cuda()
            if self.args.social_learning:
                self.social_learning_net.cuda()
            if args.rnd:
                self.rnd.cuda()
        if args.load:
            model_root_path = self.args.save_dir + '/' + 'agent_%d' % self.agent_id + '/'
            if os.path.exists(model_root_path):
                num = max([int(file.split("_")[0]) for file in os.listdir(model_root_path) if file.split("_")[0].isdigit()])
                if args.self_learning:
                    self.explore_reward_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_explore_net.pkl', map_location='cpu'))
                if args.social_learning:
                    self.social_learning_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_social_learning_net.pkl', map_location='cpu'))
                if args.rnd:
                    self.rnd.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_rnd_net.pkl', map_location='cpu'))
                self.eval_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_Q_Net.pkl', map_location='cpu'))
                self.input_net.load_state_dict(torch.load(model_root_path + '/' + str(num) + '_input_net.pkl', map_location='cpu'))
                print('Agent {} successfully loaded DQN_S2L-{}'.format(self.agent_id, num))

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.action_num = self.args.action_num
        self.learn_step_counter = 0
        self.training_parameters = [{'params': self.eval_net.parameters(), 'lr': self.args.lr},
                                    {'params': self.input_net.parameters(), 'lr': self.args.lr}]

        if self.args.rnd:
            self.training_parameters.append({'params': self.rnd.predictor.parameters(), 'lr': self.args.lr})

        if self.args.self_learning:
            self.training_parameters.append({'params': self.explore_reward_net.parameters(), 'lr': self.args.lr})

        if self.args.social_learning:
            self.training_parameters.append({'params': self.social_learning_net.parameters(), 'lr': self.args.lr})

        self.optimizer = torch.optim.Adam(self.training_parameters)
        self.loss_function = torch.nn.MSELoss()
        self.loss_entropy_function = torch.nn.CrossEntropyLoss(reduction='none')

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
        if self.args.social_learning:
            batch_in_view = torch.from_numpy(episode_data['in_view'][:,:,agent_id,...]).long()
        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()
            if self.args.social_learning:
                batch_in_view = batch_in_view.cuda()

        #print(batch_action.shape)
        q_eval = self.eval_net(self.input_net(batch_state)).gather(2, batch_action)
        q_next = self.target_net(self.input_net(batch_next_state)).detach()
        if self.learn_step_counter % 10 == 0:
            print(self.eval_net(self.input_net(batch_state))[0], self.target_net(self.input_net(batch_next_state))[0])

        reward = copy.deepcopy(batch_reward)

        if self.args.rnd:
            batch, step, H, W, C = batch_next_state.shape[0], batch_next_state.shape[1], batch_next_state.shape[2], \
                                   batch_next_state.shape[3], batch_next_state.shape[4]
            rnd_input = batch_next_state.reshape(batch * step, H, W, C)
            rnd_input = rnd_input.permute(0, 3, 1, 2)

        with torch.no_grad():
            if self.args.rnd:
                target_next_feature = self.rnd.target(rnd_input)
                predict_next_feature = self.rnd.predictor(rnd_input)
                rnd_reward = (target_next_feature - predict_next_feature).pow(2).mean(dim=1)
                rnd_reward = rnd_reward.reshape(batch, step, 1)
                rnd_reward = torch.clamp(rnd_reward, -0.25, 0.25)
                if self.learn_step_counter % 10 == 0:
                    print("env_reward:{}, rnd_reward:{}".format(batch_reward[0].mean(), rnd_reward[0].mean()))
                reward += self.args.shaping_rewards_alpha * rnd_reward

            if self.args.self_learning:
                shaping_reward = self.explore_reward_net(self.input_net(batch_state)).gather(2, batch_action)
                #shaping_reward = torch.clamp(shaping_reward / 20, -0.25, 0.25)
                if self.learn_step_counter % 10 == 0:
                    print("env_reward:{}, self_learning_reward:{}".format(batch_reward[0].mean(), shaping_reward[0].mean()))
                reward += self.args.shaping_rewards_alpha * shaping_reward
        if self.args.double_dqn:
            q_target = reward + self.args.gamma * q_next.gather(2, self.eval_net(self.input_net(batch_next_state)).max(2)[1].unsqueeze(dim=2))
        else:
            q_target = reward + self.args.gamma * q_next.max(2)[0].unsqueeze(dim=2)
        q_target = q_target.detach()

        loss_q = self.loss_function(q_eval, q_target)
        if self.args.rnd:
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
            loss = loss_q + forward_loss
        else:
            loss = loss_q
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()

        if self.args.social_learning:
            predict_social_next_state, predict_others_next_action = self.social_learning_net(self.input_net(batch_state), batch_action)
            social_loss1 = self.loss_function(predict_social_next_state, batch_next_state)
            others_next_action = torch.from_numpy(episode_data['u']).long()
            index = torch.from_numpy(np.delete(np.arange(self.args.num_agents), agent_id))
            if self.args.cuda:
                others_next_action = others_next_action.cuda()
                index = index.cuda()
            others_next_action = torch.index_select(others_next_action, 2, index)
            batch_social, step_social = predict_others_next_action[:,0:-1,...].shape[0],predict_others_next_action[:,0:-1,...].shape[1]
            a1 = predict_others_next_action[:,0:-1,...].reshape(batch_social*step_social*(self.args.num_agents-1), -1)
            a2 = others_next_action[:,1:,...].reshape(batch_social*step_social*(self.args.num_agents-1))
            social_loss2 = self.loss_entropy_function(a1, a2)*batch_in_view[:,0:-1,...].reshape(-1)
            social_loss2 = social_loss2.mean()
            social_loss = social_loss1 + 0.01*social_loss2
            print("social loss1: {}, social loss2: {}".format(social_loss1, social_loss2))
            self.optimizer.zero_grad()
            social_loss.backward()
            self.optimizer.step()

        if self.args.self_learning and self.learn_step_counter % 1 == 0:
            batch_return = torch.zeros_like(batch_reward)
            if self.args.cuda:
                batch_return = batch_return.cuda()
            for step in range(self.args.num_steps - 1, -1, -1):
                if step == self.args.num_steps - 1:
                    batch_return[:, step, ...] = batch_reward[:, step, ...]
                if step < self.args.num_steps - 1:
                    batch_return[:, step, ...] = batch_reward[:, step, ...] + self.args.gamma * batch_return[:, step + 1, ...]

            explore_reward_sa = self.explore_reward_net(self.input_net(batch_state)).gather(2, batch_action)
            explore_reward_sa_other_pi = self.explore_reward_net(self.input_net(batch_state))
            sa_other_pi = F.softmax(self.eval_net(self.input_net(batch_state))/0.1, dim=-1).detach()
            explore_reward_sa_other = torch.sum(explore_reward_sa_other_pi*sa_other_pi, dim=-1).unsqueeze(dim=2)
            loss_explore = -torch.mean(batch_return*(explore_reward_sa-explore_reward_sa_other))
            self.optimizer.zero_grad()
            loss_explore.backward()
            #torch.nn.utils.clip_grad_norm_(self.input_net.parameters(), self.args.grad_norm_clip)
            #torch.nn.utils.clip_grad_norm_(self.explore_reward_net.parameters(), self.args.grad_norm_clip)
            self.optimizer.step()

        loss_all = {}
        loss_all["q_net"] = loss_q
        if self.args.rnd:
            loss_all["rnd"] = forward_loss
        if self.args.self_learning:
            loss_all["self_learning"] = loss_explore
        if self.args.social_learning:
            loss_all["social_learning"] = social_loss
        return loss_all

    def save_model(self,num):
        model_path = self.args.save_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.eval_net.state_dict(), model_path + '/' + str(num) + '_Q_Net.pkl')
        torch.save(self.input_net.state_dict(), model_path + '/' + str(num) + '_input_net.pkl')
        if self.args.self_learning:
            torch.save(self.explore_reward_net.state_dict(), model_path + '/' + str(num) + '_explore_net.pkl')
        if self.args.rnd:
            torch.save(self.rnd.state_dict(), model_path + '/' + str(num) + '_rnd_net.pkl')
        if self.args.social_learning:
            torch.save(self.social_learning_net.state_dict(), model_path + '/' + str(num) + '_social_learning_net.pkl')
