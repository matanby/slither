import pickle
import time
import datetime
import multiprocessing as mp
import queue
import os
import argparse
import sys
import gzip

import numpy as np
import scipy.signal as ss

from policies import *

THE_DEATH_PENALTY = -100
EMPTY_VAL = 0
OBSTACLE_VAL = 100
SYMBOLS = {EMPTY_VAL: ' ',
           OBSTACLE_VAL: '+'}
MAX_PLAYERS = 20


def clear_q(q):
    while not q.empty():
        try: q.get_nowait()
        except queue.Empty: break


def days_hours_minutes_seconds(td):
    return td.days, td.seconds//3600, (td.seconds//60)%60, td.seconds%60


def random_parition(num, max_part_size):
    parts = []
    while num > 0:
        parts.append(np.random.randint(1, min(max_part_size, num+1)))
        num -= parts[-1]
    return parts


class Position(tuple):

    def __new__(cls, row, col=None):
        if col is None: row, col = row
        return super(Position, cls).__new__(cls, [row, col])

    def __add__(self, other):
        return Position(self[0] + other[0], self[1] + other[1])

    def __mod__(self, other):
        return Position(self[0] % other[0], self[1] % other[1])

    def move(self, dir):
        if dir == 'E': return self + (0,1)
        if dir == 'W': return self + (0,-1)
        if dir == 'N': return self + (-1, 0)
        if dir == 'S': return self + (1, 0)
        raise ValueError('unrecognized direction')


class Player(object):
    SHUTDOWN_TIMEOUT = 15 #seconds until policy is considered unresponsive

    def __init__(self, id, policy, policy_args, board_size, logq):
        """
        Construct a new player
        :param id: the player id
        :param policy: the class of the policy to be used by the player
        :param policy_args: string (name, value) pairs that the policy can parse to arguments
        :param board_size: the size of the game board (height, width)
        :param logq: a queue for message logging through the game
        """
        self.id = id
        self.chain = []
        self.len = 0
        self.was_initialized = False
        self.cummulative_reward = 0
        self.policy_class = policy

        self.sq = mp.Queue()
        self.aq = mp.Queue()
        self.policy = policy(policy_args, board_size, self.sq, self.aq, logq, id)
        self.policy.daemon = True
        self.policy.start()

    def init_state(self, pos, idir, size=3):
        self.chain = []
        self.chain.append(pos)
        self.chain.append(pos.move(idir))
        self.dir = idir
        self.len = 2
        self.growing = size
        self.was_initialized = True
        return self # for convenient construction

    def move(self, action='CN'):
        """
        Execute a move
        :param action: action to perform, in {'CC', 'CW', 'CN'}
        :return: a pair of positions (to_clear, to_fill). where to clear can be None
        """
        if not self.was_initialized: raise ValueError('Player was not initialized')
        (r1, c1), (r2, c2) = self.chain[:2]
        if r1 == r2 and c1 == c2:  # if tail corner merged in previous move, remove it
            self.chain = self.chain[1:]
            (r1, c1), (r2, c2) = self.chain[:2]

        to_clear = None
        if self.growing > 0:
            self.growing -= 1
            self.len += 1
        else:
            # tail movement
            to_clear = self.chain[0]
            if r1 == r2:
                if c1 < c2:
                    self.chain[0] = self.chain[0].move('E')
                else:
                    self.chain[0] = self.chain[0].move('W')
            else:
                if r1 < r2:
                    self.chain[0] = self.chain[0].move('S')
                else:
                    self.chain[0] = self.chain[0].move('N')

        # head movement
        if action != 'CN':
            self.dir = base_policy.Policy.TURNS[self.dir][action]
            self.chain.append(Position(self.chain[-1]))
        self.chain[-1] = self.chain[-1].move(self.dir)
        to_fill = self.chain[-1]

        return to_clear, to_fill

    def all_positions(self):
        pos = []
        for (r1, c1), (r2, c2) in zip(self.chain[:-1], self.chain[1:]):
            if r1 == r2:
                d = (-1) ** (1 - (c1 < c2))
                pos.extend(Position(r1, c) for c in range(c1, c2, d))
            if c1 == c2:
                d = (-1) ** (1 - (r1 < r2))
                pos.extend(Position(r, c1) for r in range(r1, r2, d))
        pos.append(self.chain[-1])
        return pos

    def state(self):
        return {'dir':self.dir, 'chain': self.chain}

    def handle_state(self, round, state, reward):
        # remove previous states from queue if they weren't handled
        if reward is not None:
            self.cummulative_reward += reward
        clear_q(self.sq)
        self.sq.put((round, state, self.state(), reward))

    def get_action(self):
        try: action = self.aq.get_nowait()
        except queue.Empty: action = base_policy.Policy.DEFAULT_ACTION
        clear_q(self.aq)
        return action

    def shutdown(self):
        clear_q(self.sq)
        self.sq.put(['get_state'])
        clear_q(self.aq)
        try:
            state = self.aq.get(timeout=Player.SHUTDOWN_TIMEOUT)
            self.sq.put(None) #shutdown signal
        except queue.Empty:
            state = None #policy is most probably dead
        self.policy.join()
        return state


class Game(object):

    @staticmethod
    def log(q, file_name, on_screen=False):
        start_time = datetime.datetime.now()
        logfile = None
        if file_name:
            logfile = gzip.GzipFile(file_name, 'w') if file_name.endswith('.gz') else open(file_name, 'wb')
        for frm, type, msg in iter(q.get, None):
            td = datetime.datetime.now() - start_time
            msg = '%i::%i:%i:%i\t%s\t%s\t%s' % (days_hours_minutes_seconds(td) + (frm, type, msg))
            if logfile: logfile.write((msg + '\n').encode('ascii'))
            if on_screen: print(msg)
        if logfile: logfile.close()

    def __init__(self, args):
        self.__dict__.update(args.__dict__)
        self.n = len(self.policies)
        assert self.n < MAX_PLAYERS, "number of players mustn't exceed $i" % MAX_PLAYERS

        # init board
        self.item_count = 0
        self.state = np.zeros(self.board_size, dtype=int)
        self.state[:] = EMPTY_VAL
        n_obs = self.obstacle_density * np.prod(self.board_size)
        # sample number of partitions
        for s in random_parition(n_obs, min(*self.board_size)):
            if np.random.rand(1) > 0.5:
                pos = self._find_empty_slot(shape=(s, 1))
                for i in range(s):
                    self.state[(pos + (i, 0)) % self.board_size] = OBSTACLE_VAL
            else:
                pos = self._find_empty_slot(shape=(1, s))
                for i in range(s):
                    self.state[(pos + (0, i)) % self.board_size] = OBSTACLE_VAL
            self.item_count += s

        self.round = 0
        self.players_init_data = [(None, None)] * len(self.policies)
        self.is_playbeack = False
        if self.playback_from is not None:
            archive = open(self.playback_from, 'rb')
            self.__dict__ = pickle.load(archive)
            self.archive = archive
            self.is_playbeack = True
            self.record_to = None
            self.to_render = args.to_render

        # init logger
        self.logq = mp.Queue()
        to_screen = not self.to_render and self.silence_log
        self.logger = mp.Process(target=self.log, args=(self.logq, self.log_file, to_screen))
        self.logger.start()

        # init player resources
        self.rewards, self.players, self.scores = [None], [None], [None]
        for i, (policy, pargs) in enumerate(self.policies):
            self.rewards.append(0)
            self.players.append(Player(i + 1, policy, pargs, self.board_size, self.logq))
            self.init_player(i + 1, *self.players_init_data[i])
            self.scores.append(0)

        # configure symbols for rendering
        self.render_map = {p.id: chr(ord('a') - 1 + p.id) for p in self.players[1:]}
        self.render_map.update(SYMBOLS)
        self.render_map.update(self.food_render_map)

        # finally, if it's a recording, then record
        if self.record_to is not None and self.playback_from is None:
            self.archive = open(self.record_to, 'wb')
            dict = self.__dict__.copy()
            del dict['players'] #remove problematic objects that are irrelevant to playback.
            del dict['archive']
            del dict['logq']
            del dict['logger']
            pickle.dump(dict, self.archive)
        self.record = self.record_to is not None

    def _find_empty_slot(self, shape=(1,3)):
        is_empty = np.asarray(self.state == EMPTY_VAL, dtype=int)
        match = ss.convolve2d(is_empty, np.ones(shape), mode='same') == np.prod(shape)
        if not np.any(match): raise ValueError('no empty slots of requested shape')
        r = np.random.choice(np.nonzero(np.any(match,axis=1))[0])
        c = np.random.choice(np.nonzero(match[r,:])[0])
        return Position(r, c)

    def reset_player(self, p):
        idx = p.all_positions()
        idx = idx[:-1] # head couldn't move forward, hence the player is initialized, so ignoring if when clearing
        self.state[tuple(zip(*[i % self.board_size for i in idx]))] = EMPTY_VAL

        # turn parts of the corpse into food TODO: for some reason food is only on diagonal - some indexing issue
        food_n = np.random.binomial(len(idx),self.food_ratio)
        if self.item_count + food_n < self.max_item_density * np.prod(self.board_size):
            subidx = np.random.choice(len(idx), size=food_n, replace=False)
            subidx = tuple(zip(*[idx[i] % self.board_size for i in subidx]))
            if subidx:
                randfood = np.random.choice(list(self.food_value_map.keys()), food_n)
                self.state[subidx] = randfood
            self.item_count += food_n
        self.init_player(p.id)

    def init_player(self, id, new_dir=None, new_pos=None):
        if new_dir is None:
            new_dir = np.random.choice(list(base_policy.Policy.TURNS.keys()))
        if new_pos is None:
            shape = (1,3) if new_dir in ['W','E'] else (3,1)
            new_pos = self._find_empty_slot(shape)
        self.players_init_data[id-1] = (new_dir, new_pos)
        p = self.players[id].init_state(new_pos, new_dir, self.init_player_size)
        self.state[new_pos] = id
        self.state[new_pos.move(p.dir)] = id
        return p

    def interact(self, move_to):
        """
        After players declared their movement, this function calculates all interactions of players with environment,
        who needs to be killed, who got what reward, and advances the remaining players.

        :param move_to: the position into which each player is moving
        :return: a list of players to reset
        """
        to_reset = []
        self.rewards[:] = [0 for _ in self.rewards]
        for p, mv in zip(self.players[1:], move_to):
            dest_val = self.state[mv]
            if dest_val != EMPTY_VAL and dest_val not in self.food_value_map:
                to_reset.append(p)
                self.rewards[p.id] = THE_DEATH_PENALTY
                continue
            self.rewards[p.id] = p.len  # you are rewarded for your length
            if dest_val in self.food_value_map:
                self.rewards[p.id] += self.food_reward_map[dest_val]
                p.growing = max(0, self.food_value_map[dest_val])  # start growing
                self.item_count -= 1
            self.state[mv] = p.id
            #update scores
            self.scores[p.id] = self.scores[p.id]*self.score_scope + self.rewards[p.id]
        return to_reset

    def randomize(self):
        if np.random.rand(1) < self.random_food_prob:
            if self.item_count < self.max_item_density * np.prod(self.board_size):
                randfood = np.random.choice(list(self.food_value_map.keys()), 1)
                self.state[self._find_empty_slot((1, 1))] = randfood
                self.item_count += 1

    def play_a_round(self):
        pperm = np.random.permutation([(i,p) for i, p in enumerate(self.players[1:])])
        # distribute states and rewards on previous round
        for i, p in pperm:
            p.handle_state(self.round, self.state, self.rewards[i+1])

        # wait and collect actions
        time.sleep(self.policy_wait_time)
        actions = {p: p.get_action() for _, p in pperm}

        move_to = []
        for p in self.players[1:]:
            fr, to = p.move(actions[p])
            if fr is not None: self.state[fr % self.board_size] = EMPTY_VAL # clear tails
            move_to.append(to % self.board_size)

        # interact with environment, and reset dead players
        to_reset = self.interact(move_to)
        self.randomize()
        for p in to_reset: self.reset_player(p)
        self.round += 1

    def render(self):
        print(chr(27)+"[2J") # ANSI clear screen
        print(' '.join(str(int(s)) for s in self.scores[1:]))
        # print(' '.join(str(p.cummulative_reward/(self.round+1)) for p in self.players[1:]))
        horzline = '-' * (self.state.shape[1] + 2)
        board = [horzline]
        for r in range(self.state.shape[0]):
            board.append('|'+''.join(self.render_map[self.state[r,c]] for c in range(self.state.shape[1]))+'|')
        board.append(horzline)
        print('\n'.join(board))

    def run(self):
        try:
            r = 0
            while r < self.game_duration:
                r += 1
                if self.to_render:
                    self.render()
                    time.sleep(self.render_rate)
                if self.is_playbeack:
                    try:
                        idx, vals = pickle.load(self.archive)
                        self.state[idx] = vals
                    except EOFError:
                        break
                else:
                    if self.record: prev = self.state.copy()
                    self.play_a_round()
                    if self.record:
                        idx = np.nonzero(self.state - prev != 0)
                        pickle.dump((idx, self.state[idx]), self.archive)
        finally:
            output = [','.join(['game_id','player_id','policy','score','state_file_path'])]
            game_id = str(abs(id(self)))
            for p, s in zip(self.players[1:], self.scores[1:]):
                state = p.shutdown()
                pstr = str(p.policy).split('<')[1].split('(')[0]
                oi = [game_id, str(p.id), pstr, str(s)]
                if self.state_folder:
                    name = '.'.join(oi[:-1])
                    state_file_path = os.path.abspath(self.state_folder) + os.path.sep + name + '.state.pkl'
                    pickle.dump(state, open(state_file_path, 'wb'))
                else:
                    state_file_path = ''
                output.append(','.join(oi + [state_file_path]))

            with open(self.output_file, 'w') as outfile:
                outfile.write('\n'.join(output))
            self.logq.put(None)
            self.logger.join()


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_argument_group('I/O')
    g.add_argument('--record_to', '-rt', type=str, default=None, help="file path to which game will be recorded.")
    g.add_argument('--playback_from', '-p', type=str, default=None,
                   help='file path from which game will be played-back (overrides record_to)')
    g.add_argument('--log_file', '-l', type=str, default=None,
                   help="a path to which game events are logged. default: game.log")
    g.add_argument('--output_file', '-o', type=str, default=None,
                   help="a path to a file in which game results and policy final states are written.")
    g.add_argument('--state_folder', '-sf', type=str, default=None,
                   help="a folder to which policies may record their states'. default: this file's folder, .\states\ ")
    g.add_argument('--to_render', '-r', action="store_false", help="whether game should not be rendered")
    g.add_argument('--render_rate', '-rr', type=float, default=0.1,
                   help='frames per second, note that the policy_wait_time bounds on the rate')
    g.add_argument('--silence_log', '-sl', action="store_false",
                   help="if rendering is off, whether game log should also not be written to the screen")

    g = p.add_argument_group('Game')
    g.add_argument('--board_size', '-bs', type=str, default='(20,40)', help='a tuple of (height, width)')
    g.add_argument('--obstacle_density', '-od', type=float, default=.02, help='the density of obstacles on the board')
    g.add_argument('--policy_wait_time', '-pwt', type=float, default=.001,
                   help='seconds to wait for policies to respond with actions')
    g.add_argument('--food_map', '-fm', type=str, default='(#,10,+1);($,20,+2);(%,50,+5)',
                   help='food icons and their respective reward, and growth effect')
    g.add_argument('--random_food_prob', '-fp', type=float, default=.01,
                   help='probability of a random food appearing in a round')
    g.add_argument('--max_item_density', '-mid', type=float, default=.2,
                   help='maximum item density in the board (not including the players)')
    g.add_argument('--food_ratio', '-fr', type=float, default=.05,
                   help='the ratio between a corpse and the number of food items it produces')
    g.add_argument('--death_penalty', '-d', type=float, default=THE_DEATH_PENALTY, help='the penalty for dying')
    g.add_argument('--game_duration', '-D', type=int, default=sys.maxsize, help='# rounds in game')

    g = p.add_argument_group('Players')
    g.add_argument('--score_scope', '-s', type=float, default=0.9998,
                   help='score is sum_i(scope^(N-i)*reward(i)), i.e. weighted sum of last ~1/(1-score_scope) elements')
    g.add_argument('--init_player_size', '-is', type=int, default=5, help='player length at start, minimum is 3')
    g.add_argument('--min_n_players', '-m', type=int, default=5, help='Minimum number of players.')
    g.add_argument('--policies', '-P', type=str, default=None,
                   help='a string describing the policies to be used in the game, of the form: '
                        '<policy_name>(<arg=val>,*);+.\n'
                        'e.g. MyPolicy(layer1=100,layer2=20);YourPolicy(your_params=123)')
    g.add_argument('--default_policy', '-dp', type=str, default='AvoidCollisions()')


    args = p.parse_args()

    # set defaults
    code_path = os.path.split(os.path.abspath(__file__))[0] + os.path.sep
    # if args.record_to is None:
    #     args.__dict__['record_to'] = code_path + 'last_game.dat'
    if not args.record_to:
        args.__dict__['record_to'] = None
    # if args.log_file is None:
    #     args.__dict__['log_file'] = code_path + 'game.log'
    if args.state_folder is None:
        args.__dict__['state_folder'] = code_path + 'states'
    if not os.path.exists(args.state_folder):
        os.mkdir(args.state_folder)
    if args.output_file is None:
        args.__dict__['output_file'] = code_path + 'game.out'

    args.__dict__['board_size'] = [int(x) for x in args.board_size[1:-1].split(',')]
    fm, fr, fv, fi = {}, {}, {}, -1
    for s in args.food_map.split(';'):
        sym, reward, val = s[1:-1].split(',')
        fm[fi], fv[fi], fr[fi], fi = sym, int(val), int(reward), fi - 1
    del args.__dict__['food_map']
    args.__dict__['food_render_map'] = fm
    args.__dict__['food_value_map'] = fv
    args.__dict__['food_reward_map'] = fr
    plcs = []
    if args.policies is not None: plcs.extend(args.policies.split(';'))
    for _ in range(len(plcs), args.min_n_players): plcs.append(args.default_policy)
    args.__dict__['policies'] = [base_policy.build(p) for p in plcs]

    args.__dict__['init_player_size'] = args.init_player_size - 2

    return args


if __name__ == '__main__':
    g = Game(parse_args())
    g.run()

