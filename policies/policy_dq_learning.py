import numpy as np
import tensorflow as tf
import time

from policies import base_policy as bp


# noinspection PyAttributeOutsideInit
class DeepQLearningPolicy(bp.Policy):
    LEARNING_RATE = 0.5
    EXPLORATION_PROB = 0.01
    GAMMA = 0.9

    DIRECTIONS_TO_IDX = {
        'N': 0,
        'S': 1,
        'E': 2,
        'W': 3,
    }

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        self._r_sum = 0

        # Keep history of states, actions and rewards
        self._states = {}
        self._actions = {}
        self._rewards = {}

        # Init TensorFlow NN:
        tf.reset_default_graph()
        self._num_inputs = self.board_size[0] * self.board_size[1] + 1  # +3 for direction, head_x, and head_y.
        self._num_actions = len(self.ACTIONS)  # 3 actions: left, right and straight.
        self._s = tf.placeholder(shape=[1, self._num_inputs], dtype=tf.float32)
        self._w = tf.Variable(tf.random_uniform([self._num_inputs, self._num_actions], 0, 0.01))
        self._q_out = tf.matmul(self._s, self._w)
        self._q_target = tf.placeholder(shape=[1, self._num_actions], dtype=tf.float32)
        self._action = tf.argmax(self._q_out, 1)
        self._loss = tf.reduce_sum(tf.square(self._q_target - self._q_out))
        self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.LEARNING_RATE).minimize(self._loss)
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    def learn(self, reward, t):
        self.log(str(reward))

        self._rewards[t] = reward

    def act(self, t, state, player_state):
        try:
            start_time = time.time()

            direction = self.DIRECTIONS_TO_IDX[player_state['dir']]
            self._states[t] = np.concatenate(
                (state.reshape(1, self._num_inputs - 1),
                 np.array([direction], ndmin=2),

                 ), axis=1,)

            # Choose an e-greedy action.
            if np.random.rand(1) < self.EXPLORATION_PROB:
                action = np.random.randint(0, len(self.ACTIONS))
            else:
                s = self._states[t]
                action = int(self._sess.run([self._action], feed_dict={self._s: s})[0])

            self._actions[t] = action

            # TODO: make sure this is slowing down the act function too much!
            if t-1 in self._rewards:
                # Optimize our current policy function.
                self.optimize_policy(
                    s1=self._states[t-1],
                    a1=self._actions[t-1],
                    s2=self._states[t],
                    r1=self._rewards[t-1]
                )

                # Cleanup memory.
                del self._states[t-1]
                del self._actions[t-1]
                del self._rewards[t-1]

            total_time = (time.time() - start_time) * 1000

            self.log('time: %s' % str(total_time))

            return self.ACTIONS[action]

        except Exception as e:
            self.log(str(e))
            return self.ACTIONS[0]

    def get_state(self):
        return None

    def optimize_policy(self, s1, a1, s2, r1):
        # Obtain the Q values of s1 and s2 by feeding them into the network.
        q_s1 = self._sess.run([self._q_out], feed_dict={self._s: s1})[0]
        q_s2 = self._sess.run(self._q_out, feed_dict={self._s: s2})[0]

        q_target = q_s1
        q_target[0, a1] = r1 + self.GAMMA * np.max(q_s2)

        # Train the network using the target and predicted Q values.
        _, loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._s: s1, self._q_target: q_target})
        self.log('loss: %s' % loss)
