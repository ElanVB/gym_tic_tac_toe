import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TicTacToeEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.action_space = spaces.Discrete(9)
		self.observation_space = spaces.Discrete(10)

	def col_connections(self, grid):
		# check if all entries in the same column are equal
		equals = np.all(grid == grid[0, :], axis=0)

		# ignore columns that are comprised of zero
		equals = equals * (grid[0, :] != 0)

		# cast bools to int and sum
		num_col_connections = np.sum(equals)

		return num_col_connections

	def row_connections(self, grid):
		# transpose the board so that the rows and the cols swap and run cols check
		return self.col_connections(grid.T)

	def diag_connect(self, diag):
		# check if all entries in the diagonal are the same
		equals = np.all(diag == diag[0])

		# ignore if the value is zero
		equals = equals * (diag[0] != 0)

		# cast bools to int and sum
		connected = np.sum(equals)

		return connected

	def diag_connections(self, grid):
		# get the top-left to bottom-right diagonal
		diag = np.diag(grid)
		num_diag_connections = self.diag_connect(diag)

		# get the top-right to bottom-left diagonal
		diag = np.diag(np.fliplr(grid))
		num_diag_connections += self.diag_connect(diag)

		return num_diag_connections

	def num_connections(self):
		# check if game is finished
		grid = np.resize(self.state, (3, 3))

		# check if the player has connected a column
		num_col_connections = self.col_connections(grid)

		# check if the player has connected a row
		num_row_connections = self.row_connections(grid)

		# check if the player has connected a diagonal
		num_diag_connections = self.diag_connections(grid)

		# total the number of connected rows / cols / diags
		connections = num_col_connections + num_row_connections + num_diag_connections

		return connections

	def player_step(self, action, done, reward):
		# check move legality
		if self.state[action] != 0:
			self.done = done = True
			reward = -1
		else:
			# apply move
			self.state[action] = self.player

			connections = self.num_connections()

			if connections > 0:
				self.done = done = True
				reward = 1
			else:
				moves_left = self.available_actions()
				if moves_left.shape[0] == 0:
					# draw outcome
					self.done = done = True
					reward = 0

		return done, reward

	def opponent_step(self, done, reward):
		# the opponent needs to make a move
		# get random move and apply it
		moves_left = self.available_actions()
		rand_move = np.random.choice(moves_left)
		self.state[rand_move] = self.opponent

		connections = self.num_connections()

		if connections > 0:
			self.done = done = True
			reward = -1
		else:
			# check if any more moves are available
			moves_left = self.available_actions()
			if moves_left.shape[0] == 0:
				# draw outcome
				self.done = done = True
				reward = 0

		return done, reward

	def half_step(self, action, player):
		if self.done:
			raise Exception("The game has finished, call the reset funciton.")

		done = False
		reward = 0

		# check move legality
		if self.state[action] != 0:
			self.done = done = True
			reward = -1
		else:
			# apply move
			self.state[action] = player

			connections = self.num_connections()

			if connections > 0:
				self.done = done = True
				reward = 1
			else:
				moves_left = self.available_actions()
				if moves_left.shape[0] == 0:
					# draw outcome
					self.done = done = True
					reward = 0

		# normalize state
		return self.append_player_to_state() / 2, reward, done, None

	def step(self, action):
		if self.done:
			raise Exception("The game has finished, call the reset funciton.")

		done = False
		reward = 0

		done, reward = self.player_step(action, done, reward)
		if not done:
			done, reward = self.opponent_step(done, reward)

		# normalize state
		return self.append_player_to_state() / 2, reward, done, None

	def append_player_to_state(self):
		return np.append(self.state, np.array(self.player, dtype=int))

	def reset(self, player=None, two_player=False):
		self.done = False
		self.state = np.zeros(9, dtype=int)

		if not two_player:
			if player is None:
				self.player = np.random.choice([1, 2])
			else:
				self.player = player

			if self.player == 2:
				self.opponent = 1

				# since opponent is player 1, they must make the first move now
				self.opponent_step(False, 0)
			else:
				self.opponent = 2
			self.opponent = 1 if self.player == 2 else 2

		# normalize
		return self.append_player_to_state() / 2

	def render(self, mode='human'):
		state_list = np.resize(self.state, (3, 3)).astype(str).tolist()
		state_str = []

		for row in state_list:
			state_str.append(" | ".join(row))
		state_str = ("\n" + "-"*9 + "\n").join(state_str)

		print("player: {}".format(self.player))
		print(state_str)

	def available_actions(self):
		return np.arange(9)[self.state == 0]

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

