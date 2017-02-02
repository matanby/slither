import sys
import traceback
import random
import numpy as np
import time
from collections import OrderedDict

import tensorflow as tf
from policies import base_policy as bp


# noinspection PyAttributeOutsideInit
class DeepQLearningPolicy(bp.Policy):
    DEFAULT_LEARNING_RATE = 1e-2
    DEFAULT_EXPLORATION_PROB = 0.3
    MAX_MEMORY_STEPS = 1000
    MIN_EXPLORATION_PROB = 0.05
    EXPLORATION_PROP_DECAY_STEP = 1000
    DEFAULT_GAMMA = 0.5
    MINI_BATCH_SIZE = 500
    RANDOM_BATCH_SAMPLE = False
    OPTIMIZATION_STEP = 20
    CROP_SIZE = 3
    OBSERVATION_TIME = 400
    FC_NETWORK = True
    DEATH_PENALTY = -100
    N_HIDDEN1_MIN = 32
    REWORD_PERCOLATION = 1
    COMPRESS_STATE = False
    SPLIT_LAYERS = False

    DIRECTIONS_TO_IDX = {
        'N': 0,
        'S': 1,
        'E': 2,
        'W': 3,
    }

    def cast_string_args(self, policy_args):
        args = {
            'learning_rate': float(policy_args.get('lr', self.DEFAULT_LEARNING_RATE)),
            'exploration_prob': float(policy_args.get('e', self.DEFAULT_EXPLORATION_PROB)),
            'min_exploration_prob': float(policy_args.get('e_min', self.MIN_EXPLORATION_PROB)),
            'exploration_prop_decay_step': float(policy_args.get('e_ds', self.EXPLORATION_PROP_DECAY_STEP)),
            'gamma': float(policy_args.get('g', self.DEFAULT_GAMMA)),
            'mini_batch_size': int(policy_args.get('bs', self.MINI_BATCH_SIZE)),
            'crop_size': int(policy_args.get('cs', self.CROP_SIZE)),
            'heat_policy': policy_args.get('hp', self.COMPRESS_STATE) == 'True',
        }

        # Log active configuration
        for name, value in args.items():
            self.log('%s: %s' % (name, value), type='config')

        return args

    def init_run(self):
        # Keep history of states, actions and rewards
        # memory maps between a time (t) and (s_t, a_t, s_t+1, r_t)
        self._memory = OrderedDict()

        # Init a set where board items will be saved while running in observation mode.
        self._board_objects = set()
        self._board_objects_num = 1

        # Initialize a counter of the current timestamp.
        self._time = 0

        self._board_height = self.board_size[0] if self.board_size[0] % 2 == 1 else self.board_size[0] - 1
        self._board_width = self.board_size[1] if self.board_size[1] % 2 == 1 else self.board_size[1] - 1
        
        if self.COMPRESS_STATE:
            self.heat_map1 = self.create_heat_map_indicator(m=self._board_height, n=self._board_width)
            self.heat_map2 = self.create_heat_map_indicator(n=self._board_height, m=self._board_width)

        if not self.SPLIT_LAYERS:
            self._state_size = (2 * self.crop_size + 1) ** 2
            self.build_network()

    def build_network(self):
        def weight_var(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_var(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # Init TensorFlow NN:
        tf.reset_default_graph()

        self._num_actions = len(self.ACTIONS)
        self._s = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32)

        # Fully connected neural network.
        if self.FC_NETWORK:
            n_hidden1 = max(self._state_size * 2, self.N_HIDDEN1_MIN)

            # self._w1 = weight_var([self._state_size, self._num_actions])
            # self._b1 = bias_var([self._num_actions])
            # self._q_out = tf.matmul(self._s, self._w1) + self._b1

            self._w1 = weight_var([self._state_size, n_hidden1])
            self._b1 = bias_var([n_hidden1])
            self._h1 = tf.nn.relu(tf.matmul(self._s, self._w1) + self._b1)

            self._w2 = weight_var([n_hidden1, self._num_actions])
            self._b2 = bias_var([self._num_actions])
            self._q_out = tf.matmul(self._h1, self._w2) + self._b2

        # Convolutional neural network.
        else:
            def conv2d(x, W):
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            s_reshaped = tf.reshape(self._s, [-1, 2 * self.crop_size + 1, 2 * self.crop_size + 1, self._board_objects_num])

            w_conv1 = weight_var([2, 2, self._board_objects_num, 3])
            b_conv1 = bias_var([3])

            h_conv1 = tf.nn.relu(conv2d(s_reshaped, w_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            w_conv2 = weight_var([2, 2, 3, 6])
            b_conv2 = bias_var([6])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            h_pool2_shape = h_pool2.get_shape()
            new_dim = int(h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3])

            w_fc1 = weight_var([new_dim, self._num_actions])
            b_fc1 = bias_var([self._num_actions])

            h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim])
            self._q_out = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

            # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            # W_fc2 = weight_var([256, self._num_actions])
            # b_fc2 = bias_var([self._num_actions])
            # self._q_out = tf.matmul(h_fc1, W_fc2) + b_fc2
            ###

        self._q_target = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        self._action = tf.argmax(self._q_out, axis=1)
        self._loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._q_target - self._q_out), reduction_indices=1))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def learn(self, reward, t):
        try:
            if self._time <= self.OBSERVATION_TIME + 1 and self.SPLIT_LAYERS:
                return

            self._memory.setdefault(t - 1, [None, None, None, None])
            self._memory[t - 1][3] = reward

            # Percolate a positive reword to the previous steps.
            if reward > 6:
                for i in range(2, self.REWORD_PERCOLATION + 1):
                    if t - i not in self._memory:
                        break

                    if self._memory[t - i][3] is None or self._memory[t - i][3] == self.DEATH_PENALTY:
                        break
                    else:
                        self._memory[t - i][3] += reward / i

            # Optimize our current policy function.
            if (self._time + 1) % self.OPTIMIZATION_STEP == 0:
                self.optimize_policy()

            if (t + 1) % self.exploration_prop_decay_step == 0 and (self.exploration_prob / 2) > self.min_exploration_prob:
                self.exploration_prob /= 2
                self.log('lowering exploration_prob to: %.5f' % self.exploration_prob)

        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.log("\n%s" % tb_str, type='error')

    def act(self, t, state, player_state):
        try:
            start_time = time.time()
            self._time += 1

            # If the observation time is over, build the network
            if self.SPLIT_LAYERS:
                if self._time < self.OBSERVATION_TIME:
                    self._board_objects |= set(state.flatten())
                    self._board_objects_num = len(self._board_objects) + 1

                    # Choose an action by trying to avoid collisions.
                    return self.avoid_collisions(state, player_state)

                elif self._time == self.OBSERVATION_TIME:
                    self.log('Finished observation time, num of board objects: %d' % self._board_objects_num)
                    self.log('Objects: %s' % str(self._board_objects))
                    if self.heat_policy:
                        self._state_size = 6 * self._board_objects_num
                    else:
                        self._state_size = (2 * self.crop_size + 1) ** 2 * self._board_objects_num
                    self.build_network()

            player_head = player_state['chain'][-1]
            player_direction = player_state['dir']
            state_norm = self.normalize_state(state, player_head[0], player_head[1], player_direction)
            
            if self.SPLIT_LAYERS:
                state_norm = self.split_layers(state_norm)

            state_vec = state_norm.reshape((1, self._state_size))

            # Choose an e-greedy action.
            if np.random.rand(1) < self.exploration_prob:
                action = np.random.randint(0, len(self.ACTIONS))
            else:
                action = int(self._sess.run([self._action], feed_dict={self._s: state_vec})[0])

            self._memory.setdefault(t, [None, None, None, None])
            self._memory[t][0] = state_vec
            self._memory[t][1] = action

            self._memory.setdefault(t - 1, [None, None, None, None])
            self._memory[t - 1][2] = state_vec

            # If our memory is full, remove the oldest item.
            if len(self._memory) > self.MAX_MEMORY_STEPS:
                self._memory.popitem(last=False)

            total_time = (time.time() - start_time) * 1000
            # self.log('total act time (ms): %.2f' % total_time)

            return self.ACTIONS[action]

        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.log("\n%s" % tb_str, type='error')
            return random.choice(self.ACTIONS)

    def avoid_collisions(self, state, player_state):
        head_pos = player_state['chain'][-1]
        a = self.ACTIONS[min(np.random.randint(20), 2)]  # 10% of actions are random
        for a in [a] + list(np.random.permutation(self.ACTIONS)):
            r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][a]) % state.shape
            if state[r, c] <= 0: return a
        return a

    def normalize_state(self, state, axis0, axis1, direction):
        axis0 = axis0 % self.board_size[0]
        axis1 = axis1 % self.board_size[1]

        shift_0 = self.board_size[0] // 2 - axis0
        shift_1 = self.board_size[1] // 2 - axis1
        state = np.roll(np.roll(state, shift_0, axis=0), shift_1, axis=1)

        if self.board_size[0] % 2 == 0:
            state = state[1:, :]

        if self.board_size[1] % 2 == 0:
            state = state[:, 1:]

        rotate = lambda x: np.array(list(reversed(list(zip(*x)))))

        if direction == 'N':
            pass
        elif direction == 'E':
            state = rotate(state)
        elif direction == 'S':
            state = rotate(rotate(state))
        elif direction == 'W':
            state = rotate(rotate(rotate(state)))

        if not self.heat_policy:
            center_0 = self._board_height // 2
            center_1 = self._board_width // 2
            state = state[
                    center_0 - self.crop_size: center_0 + self.crop_size + 1,
                    center_1 - self.crop_size: center_1 + self.crop_size + 1
            ]

        return state

    def split_layers(self, state):
        board_size = state.shape[0] * state.shape[1]
        result = np.repeat(state.reshape((board_size, 1)), self._board_objects_num).reshape((state.shape[0], state.shape[1], self._board_objects_num))

        for i, obj in enumerate(self._board_objects):
            result[state == obj, i] = 1
            result[state != obj, i] = 0
            result[state == obj, self._board_objects_num - 1] = 0

        if not self.heat_policy:
            return result

        else:
            center_0 = self._board_height // 2
            center_1 = self._board_width // 2
            features = np.zeros((self._board_objects_num * 6, 1))
            for i in range(self._board_objects_num):
                # indicators per object:
                features[i * 6] = result[center_0 - 1, center_1, i]  # up
                features[i * 6 + 1] = result[center_0, center_1 + 1, i]  # right
                features[i * 6 + 2] = result[center_0, center_1 - 1, i]  # left

                # heat map per object:
                if result[:, :, i].shape == self.heat_map1.shape:
                    temp = self.heat_map1 * result[:, :, i]
                else:
                    temp = self.heat_map2 * result[:, :, i]

                up = temp[:center_0, :].mean()
                right = temp[:, center_1 + 1:].mean()
                left = temp[:, : center_1].mean()

                features[i * 6 + 3] = 1 if up == max(up, right, left) else 0
                features[i * 6 + 4] = 1 if right == max(up, right, left) else 0
                features[i * 6 + 5] = 1 if left == max(up, right, left) else 0

            return features

    def get_state(self):
        return None

    def optimize_policy(self):
        batch_size = min(len(self._memory), self.mini_batch_size)
        if self.RANDOM_BATCH_SAMPLE:
            batch = random.sample(list(self._memory.values()), batch_size)
        else:
            batch = list(self._memory.values())[-batch_size:]

        # Remove items which are not yet complete.
        batch = [i for i in batch if None not in i]
        batch_size = len(batch)

        if not batch:
            return

        s1 = np.array([i[0] for i in batch]).reshape(len(batch), self._state_size)
        a1 = np.array([i[1] for i in batch]).reshape(len(batch), 1)
        s2 = np.array([i[2] for i in batch]).reshape(len(batch), self._state_size)
        r1 = np.array([i[3] for i in batch]).reshape(len(batch), 1)

        # Obtain the Q values of s1 and s2 by feeding them into the network.
        q_s1 = self._sess.run([self._q_out], feed_dict={self._s: s1})[0]
        q_s2 = self._sess.run([self._q_out], feed_dict={self._s: s2})[0]

        q_target = q_s1

        for i in range(batch_size):
            # q_target[i, a1[i]] = r1[i] + self.gamma * np.max(q_s2[i])

            if r1[i] <= self.DEATH_PENALTY:
                q_target[i, a1[i]] = r1[i]
            else:
                q_target[i, a1[i]] = r1[i] + self.gamma * np.max(q_s2[i])

        # Train the network using the target and predicted Q values.
        _, avg_loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._s: s1, self._q_target: q_target})
        avg_reward = np.mean(r1)
        self.log('time: %d, batch size: %s, avg reward: %.2f, avg loss: %s' % (self._time, batch_size, avg_reward, avg_loss), 'optimization')

    @staticmethod
    def create_heat_map_indicator(m, n):
        x = np.exp(-np.power(np.linspace(-np.sqrt(n), np.sqrt(n), n), 2.) / (2 * np.power(2, 2.)))
        y = np.exp(-np.power(np.linspace(-np.sqrt(m), np.sqrt(m), m), 2.) / (2 * np.power(2, 2.)))
        mesh_x, mesh_y = np.meshgrid(x, y)
        heat_map = mesh_x + mesh_y
        heat_map /= np.max(heat_map)
        heat_map **= 2
        return heat_map
