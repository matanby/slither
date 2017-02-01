import random
import numpy as np
import time
from collections import OrderedDict

import tensorflow as tf
from policies import base_policy as bp


# noinspection PyAttributeOutsideInit
class DeepQLearningPolicy(bp.Policy):
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_EXPLORATION_PROB = 0.3
    MIN_EXPLORATION_PROB = 0.02
    DEFAULT_GAMMA = 0.1
    MAX_MEMORY_STEPS = 300
    MINI_BATCH_SIZE = 200
    CROP_SIZE = 3

    DIRECTIONS_TO_IDX = {
        'N': 0,
        'S': 1,
        'E': 2,
        'W': 3,
    }

    def cast_string_args(self, policy_args):
        policy_args['learning_rate'] = float(policy_args['lr']) if 'lr' in policy_args else self.DEFAULT_LEARNING_RATE
        policy_args['exploration_prob'] = float(policy_args['e']) if 'e' in policy_args else self.DEFAULT_EXPLORATION_PROB
        policy_args['gamma'] = float(policy_args['g']) if 'g' in policy_args else self.DEFAULT_GAMMA
        return policy_args

    def init_run(self):
        # Keep history of states, actions and rewards
        # memory maps between a time (t) and (s_t, a_t, s_t+1, r_t)
        self._memory = OrderedDict()

        self._time = 0

        self.build_network()

        # Log active configuration
        self.log('learning rate: %s' % self.learning_rate)
        self.log('exploration_prob: %s' % self.exploration_prob)
        self.log('gamma: %s' % self.gamma)

    def build_network(self):
        n_hidden1 = 1024

        def weight_var(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_var(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Init TensorFlow NN:
        tf.reset_default_graph()
        self._board_height = self.board_size[0] if self.board_size[0] % 2 == 1 else self.board_size[0] - 1
        self._board_width = self.board_size[1] if self.board_size[1] % 2 == 1 else self.board_size[1] - 1
        # self._state_size = self._board_height * self._board_width
        LAYERS = 3
        self._state_size = (2 * self.CROP_SIZE + 1)**2 * LAYERS

        self._num_actions = len(self.ACTIONS)
        self._s = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32)

        # self._w1 = weight_var([self._state_size, self._num_actions])
        # self._b1 = bias_var([self._num_actions])
        # self._q_out = tf.matmul(self._s, self._w1) + self._b1

        ####
        self._w1 = weight_var([self._state_size, n_hidden1])
        self._b1 = bias_var([n_hidden1])
        self._h1 = tf.nn.relu(tf.matmul(self._s, self._w1) + self._b1)
        self._w2 = weight_var([n_hidden1, self._num_actions])
        self._b2 = bias_var([self._num_actions])
        self._q_out = tf.matmul(self._h1, self._w2) + self._b2

        ###
        # s_reshaped = tf.reshape(self._s, [-1, 2 * self.CROP_SIZE + 1, 2 * self.CROP_SIZE + 1, LAYERS])
        #
        # W_conv1 = weight_var([2, 2, LAYERS, 3])
        # b_conv1 = bias_var([3])
        #
        # h_conv1 = tf.nn.relu(conv2d(s_reshaped, W_conv1) + b_conv1)
        # h_pool1 = max_pool_2x2(h_conv1)
        #
        # W_conv2 = weight_var([2, 2, 3, 6])
        # b_conv2 = bias_var([6])
        #
        # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)
        #
        # h_pool2_shape = h_pool2.get_shape()
        # new_dim = int(h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3])
        #
        # W_fc1 = weight_var([new_dim, self._num_actions])
        # b_fc1 = bias_var([self._num_actions])
        #
        # h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim])
        # self._q_out = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # W_fc2 = weight_var([256, self._num_actions])
        # b_fc2 = bias_var([self._num_actions])
        # self._q_out = tf.matmul(h_fc1, W_fc2) + b_fc2
        ###

        self._q_target = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        self._action = tf.argmax(self._q_out, axis=1)
        self._loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._q_target - self._q_out), reduction_indices=1))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.DEFAULT_LEARNING_RATE).minimize(self._loss)
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    def learn(self, reward, t):
        try:
            # self.log('time: %s, reward: %s' % (t, reward))

            self._memory.setdefault(t, [None, None, None, None])
            self._memory[t-1][3] = reward

            # Optimize our current policy function.
            if (self._time + 1) % 20 == 0:
                self.optimize_policy()

            if (t+1) % 1000 == 0 and self.exploration_prob > self.MIN_EXPLORATION_PROB:
                self.exploration_prob /= 2
                self.log('lowering exploration_prob to: %.5f' % self.exploration_prob)

        except Exception as e:
            self.log('%s: %s' % (type(e), str(e)))

    def act(self, t, state, player_state):
        try:
            start_time = time.time()
            self._time += 1

            player_head = player_state['chain'][-1]
            player_direction = player_state['dir']
            state_norm = self.normalize_state(state, player_head[0], player_head[1], player_direction)
            state_norm = self.split_layers(state_norm)
            state_vec = state_norm.reshape((1, self._state_size))

            # self._previous_state = state_norm
            # self._previous_direction = player_direction
            # self._previous_chain = player_state['chain']
            # self.log('direction: %s, chain: %s' % (player_direction, player_state['chain']))

            # direction = self.DIRECTIONS_TO_IDX[player_state['dir']]
            # state_vec = np.concatenate((
            #     state.reshape(1, self._state_size - 5),
            #     np.array([direction], ndmin=2),
            #     np.array([player_state['chain'][0][0], player_state['chain'][0][1]], ndmin=2),
            #     np.array([player_state['chain'][-1][0], player_state['chain'][-1][1]], ndmin=2),
            # ), axis=1,)

            # Choose an e-greedy action.
            if np.random.rand(1) < self.exploration_prob:
                action = np.random.randint(0, len(self.ACTIONS))
            else:
                action = int(self._sess.run([self._action], feed_dict={self._s: state_vec})[0])

            #####
            if self._time % 20 == 0:
                q_out = self._sess.run([self._q_out], feed_dict={self._s: state_vec})[0]
                self.log('%s' % q_out)

            self._memory.setdefault(t, [None, None, None, None])
            self._memory[t][0] = state_vec
            self._memory[t][1] = action

            self._memory.setdefault(t-1, [None, None, None, None])
            self._memory[t-1][2] = state_vec

            # If our memory is full, remove the oldest item.
            if len(self._memory) > self.MAX_MEMORY_STEPS:
                self._memory.popitem(last=False)

            # TODO: make sure this is slowing down the act function too much!
            total_time = (time.time() - start_time) * 1000
            # self.log('total act time (ms): %.2f' % total_time)

            return self.ACTIONS[action]

        except Exception as e:
            self.log('%s: %s' % (type(e), str(e)), 'error')
            return random.choice(self.ACTIONS)

    def split_layers(self, state):
        total = state.shape[0] * state.shape[1]
        result = np.repeat(state.reshape((total, 1)), 3).reshape((state.shape[0], state.shape[1], 3))

        result[state == 1, 0] = 1
        result[state != 1, 0] = 0

        result[state == -1, 1] = 1
        result[state != -1, 1] = 0

        result[state == 100, 2] = 1
        result[state != 100, 2] = 0

        return result

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

        center_0 = self._board_height // 2
        center_1 = self._board_width // 2
        state = state[
                center_0 - self.CROP_SIZE: center_0 + self.CROP_SIZE + 1,
                center_1 - self.CROP_SIZE: center_1 + self.CROP_SIZE + 1
        ]

        return state

    def get_state(self):
        # TODO: implement.
        return None

    def optimize_policy(self):
        batch_size = min(len(self._memory), self.MINI_BATCH_SIZE)
        # batch = random.sample(list(self._memory.values()), batch_size)
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
        # q_target[0, a1] = r1 + self.gamma * np.max(q_s2, axis=1).reshape((batch_size, 1))
        for i in range(batch_size):
            # q_target[i, a1[i]] = r1[i] + self.gamma * np.max(q_s2[i])

            if r1[i] <= -100:
                q_target[i, a1[i]] = r1[i]
            else:
                q_target[i, a1[i]] = r1[i] + self.gamma * np.max(q_s2[i])

        # Train the network using the target and predicted Q values.
        _, avg_loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._s: s1, self._q_target: q_target})
        avg_reward = np.mean(r1)
        self.log('time: %d, batch size: %s, avg reward: %.2f, avg loss: %s' % (self._time, batch_size, avg_reward, avg_loss), 'optimization')
