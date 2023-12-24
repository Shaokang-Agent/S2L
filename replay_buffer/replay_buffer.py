import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, args):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.args = args
        if (args.algorithm == "DQN" or args.algorithm == "DQN_EXPLORE_S2L") and args.num_agents > 1 and args.social_learning:
            self._storage = {'o': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, self.args.state_shape, self.args.state_shape, 3]),
                            'u': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'r': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'o_next': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, self.args.state_shape, self.args.state_shape, 3]),
                            'in_view': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, self.args.num_agents-1]),
                            }
        else:
            self._storage = {'o': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, self.args.state_shape, self.args.state_shape, 3]),
                            'u': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'r': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'o_next': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, self.args.state_shape, self.args.state_shape, 3]),
                            }
        self.actual_length = 0
        self.index = 0

    def __len__(self):
        return self.actual_length

    def add(self, episode_data):
        self.actual_length = min(self.args.buffer_size, self.actual_length+1)
        if "o" in episode_data.keys():
            self._storage['o'][self.index] = np.array(episode_data['o'])
        if "u" in episode_data.keys():
            self._storage['u'][self.index] = np.expand_dims(np.array(episode_data['u']), axis=2)
        if "r" in episode_data.keys():
            self._storage['r'][self.index] = np.expand_dims(np.array(episode_data['r']), axis=2)
        if "o_next" in episode_data.keys():
            self._storage['o_next'][self.index] = np.array(episode_data['o_next'])
        if "in_view" in episode_data.keys():
            self._storage['in_view'][self.index] = np.array(episode_data['in_view'])
        self.index = (self.index + 1) % self.args.buffer_size

    def _encode_sample(self, idxes):
        temp_buffer = {}
        for key in self._storage.keys():
            temp_buffer[key] = self._storage[key][idxes]
        return temp_buffer

    def sample(self, batch_size):
        sample_list = [i for i in range(self.actual_length)]
        if self.actual_length < 1000:
            idxes = random.sample(sample_list, batch_size)
        else:
            idxes = [random.randint(0, self.actual_length-1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
