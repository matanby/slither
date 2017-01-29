import sys
import multiprocessing as mp

import numpy as np
np.set_printoptions(threshold=np.nan)

GAMMA = .5
RATE = .01
EPSILON = .01
BATCH = 20

POLICIES = {}


def discount(r, gamma=1.0):
    """ compute discounted reward for a reward vector r with discount factor gamma"""
    dr = np.zeros(r.shape)
    sum_from_end = 0
    for i in range(len(r)-1,-1,-1):
        sum_from_end = r[i] + sum_from_end * gamma
        dr[i] = sum_from_end
    return dr


def collect_policies():
    if POLICIES: return POLICIES # only fill on first function call
    for mname in sys.modules:
        if not mname.startswith('policies.policy'): continue
        mod = sys.modules[mname]
        for cls_name in dir(mod):
            try:
                if cls_name != 'Policy':
                    cls = mod.__dict__[cls_name]
                    if issubclass(cls, Policy): POLICIES[cls_name] = cls
            except TypeError:
                pass
    return POLICIES


def build(policy_string):
    available_policies = collect_policies()
    name, args = policy_string.split('(')
    if name not in available_policies: raise ValueError('no such policy: %s' % name)
    P = available_policies[name]
    kwargs = dict(tuple(arg.split('=')) for arg in args[:-1].split(',') if arg)
    return P, kwargs


class Policy(mp.Process):
    DEFAULT_ACTION = 'CN'
    ACTIONS = ['CC',  # counter clockwise
               'CW',  # clockwise
               'CN']  # continue
    TURNS = {
        'N': {'CC': 'W', 'CW': 'E', 'CN': 'N'},
        'S': {'CC': 'E', 'CW': 'W', 'CN': 'S'},
        'W': {'CC': 'S', 'CW': 'N', 'CN': 'W'},
        'E': {'CC': 'N', 'CW': 'S', 'CN': 'E'}
    }

    def __init__(self, policy_args, board_size, stateq, actq, logq, id):
        mp.Process.__init__(self)
        self.sq = stateq
        self.aq = actq
        self.lq = logq
        self.id = id
        self.board_size = board_size
        self.__dict__.update(self.cast_string_args(policy_args))

    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: a arg -> string value map as received in command line, notice that the "load_from" is a
                            special argument passing a file name that contains a pickled state that can be used for
                            initialization
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        raise NotImplementedError

    def run(self):
        try:
            self.init_run()
            for input in iter(self.sq.get, None):
                if input[0] == 'get_state':
                    self.aq.put(self.get_state())
                else:
                    t, state, player_state, reward = input
                    self.learn(reward, t)
                    self.aq.put(self.act(t, state, player_state))
        except:
            self.log( "policy %s is down." % str(self), type='error')
            for input in iter(self.sq.get, None):
                if input[0] == 'get_state': self.aq.put(None)

    def init_run(self):
        """
        initialize vars that are not primitives (e.g. TF session)
        """
        raise NotImplementedError

    def learn(self, reward, t):
        """
        :param t: the time, in case timesteps are missed
        :param reward: the reward from time=t-1
        """
        raise NotImplementedError

    def act(self, t, state, player_state):
        """
        :param t: the time, in case timesteps are missed
        :param state: the game board
        :param player_state: a tuple of (position, direction)
        :return: A single action from Policy.Actions
        """
        raise NotImplementedError

    def log(self, msg, type=' '):
        self.lq.put((str(self.id), type, msg))

    def get_state(self):
        """
        :return: the current policy state (e.g. for future initialization)
        """
        raise NotImplementedError

