from policies import base_policy as bp
import numpy as np
import pickle


class Policy2354678(bp.Policy):

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        try:
            state = pickle.load(open(self.load_from))
        except IOError:
            state = np.zeros(100)
        self.state = state

    def learn(self, reward, t):
        pass

    def act(self, t, state, player_state):
        raise ValueError()

    def get_state(self):
        return self.state