import random
import numpy as np
import time
from collections import OrderedDict

import tensorflow as tf
from policies import base_policy as bp


# noinspection PyAttributeOutsideInit
class DeepQLearningPolicy(bp.Policy):
    DEFAULT_LEARNING_RATE = 0.0000001
    DEFAULT_EXPLORATION_PROB = 0.1
    DEFAULT_GAMMA = 0.01
    MAX_MEMORY_STEPS = 100
    MINI_BATCH_SIZE = 25

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

        self.build_network()

        # Log active configuration
        self.log('learning rate: %s' % self.learning_rate)
        self.log('exploration_prob: %s' % self.exploration_prob)
        self.log('gamma: %s' % self.gamma)

    def build_network(self):
        n_hidden1 = 512

        def weight_var(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

        def bias_var(shape):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

        # Init TensorFlow NN:
        tf.reset_default_graph()
        board_height = self.board_size[0] if self.board_size[0] % 2 == 1 else self.board_size[0] - 1
        board_width = self.board_size[1] if self.board_size[1] % 2 == 1 else self.board_size[1] - 1
        self._state_size = board_height * board_width

        self._num_actions = len(self.ACTIONS)
        self._s = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32)
        self._w1 = weight_var([self._state_size, self._num_actions])
        self._b1 = bias_var([self._num_actions])
        # self._w1 = weight_var([self._state_size, n_hidden1])
        # self._b1 = bias_var([n_hidden1])
        # self._h1 = tf.nn.relu(tf.matmul(self._s, self._w1) + self._b1)
        # self._w2 = weight_var([n_hidden1, self._num_actions])
        # self._b2 = bias_var([self._num_actions])
        # self._q_out = tf.matmul(self._h1, self._w2) + self._b2
        self._q_out = tf.matmul(self._s, self._w1) + self._b1
        self._q_target = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        self._action = tf.argmax(self._q_out, axis=1)
        self._loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._q_target - self._q_out), reduction_indices=1))
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.DEFAULT_LEARNING_RATE).minimize(self._loss)
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    def learn(self, reward, t):
        try:
            # self.log('time: %s, reward: %s' % (t, reward))

            self._memory.setdefault(t, [None, None, None, None])
            self._memory[t][3] = reward

            # Optimize our current policy function.
            self.optimize_policy()

            if (t+1) % 1000 == 0:
                self.log('lowering exploration_prob to: %.5f' % self.exploration_prob)
                self.exploration_prob /= 2

        except Exception as e:
            self.log('%s: %s' % (type(e), str(e)))

    def act(self, t, state, player_state):
        try:
            start_time = time.time()

            # TODO: verify head is chain[0]
            head = player_state['chain'][0]
            direction = player_state['dir']
            state_centralized = self.centralize(state, head[0], head[1])
            state_rot = self.rotate(state_centralized, direction)
            state_vec = state_rot.reshape((1, self._state_size))

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

            self._memory.setdefault(t, [None, None, None, None])
            self._memory[t][0] = state_vec
            self._memory[t][1] = action

            self._memory.setdefault(t-1, [None, None, None, None])
            self._memory[t-1][2] = state_vec

            # If our memory is full, remove the oldest item.
            if len(self._memory) > self.MAX_MEMORY_STEPS:
                self._memory.popitem(last=False)

            total_time = (time.time() - start_time) * 1000

            # TODO: make sure this is slowing down the act function too much!
            # self.log('total act time (ms): %.2f' % total_time)

            return self.ACTIONS[action]

        except Exception as e:
            self.log('%s: %s' % (type(e), str(e)), 'error')
            return random.choice(self.ACTIONS)

    def centralize(self, state, axis0, axis1):
        axis0 = axis0 % self.board_size[0]
        axis1 = axis1 % self.board_size[1]

        shift_0 = self.board_size[0] // 2 - axis0
        shift_1 = self.board_size[1] // 2 - axis1
        shifted = np.roll(np.roll(state, shift_0, axis=0), shift_1, axis=1)

        if self.board_size[0] % 2 == 0:
            shifted = shifted[1:, :]

        if self.board_size[1] % 2 == 0:
            shifted = shifted[:, 1:]

        return shifted

    def rotate(self, state, direction):
        if direction == 'N':
            return state
        elif direction == 'E':
            return np.array(list(reversed(list(zip(*state)))))
        elif direction == 'S':
            t1 = np.array(list(reversed(list(zip(*state)))))
            return np.array(list(reversed(list(zip(*t1)))))
        elif direction == 'W':
            t1 = np.array(list(reversed(list(zip(*state)))))
            t2 = np.array(list(reversed(list(zip(*t1)))))
            return np.array(list(reversed(list(zip(*t2)))))

    def get_state(self):
        # TODO: implement.
        return None

    def optimize_policy(self):
        batch_size = min(len(self._memory), self.MINI_BATCH_SIZE)
        batch = random.sample(list(self._memory.values()), batch_size)

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

            if r1[i] > 0:
                q_target[i, a1[i]] = r1[i] + self.gamma * np.max(q_s2[i])
            else:
                q_target[i, a1[i]] = r1[i]

        # Train the network using the target and predicted Q values.
        _, avg_loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._s: s1, self._q_target: q_target})
        avg_reward = np.mean(r1)
        self.log('batch size: %s, avg reward: %.2f, avg loss: %s' % (batch_size, avg_reward, avg_loss), 'optimization')
