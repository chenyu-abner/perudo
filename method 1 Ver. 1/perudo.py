import random
import sys
import numpy as np
from bet import DUDO
from AI import ComputerPlayer
from AI import RandomPlayer
from AI import TotalRandom
from strings import INSUFFICIENT_BOTS
from strings import INSUFFICIENT_DICE
import matplotlib.pyplot as plt
from tqdm import tqdm

# "Burn all you love."
bot_names = ['Winston', 'Luke', 'Jeff', 'Jia', 'Ben']

class Perudo(object):

	def __init__(self, name, player_number, dice_number):
		self.round = 0
		self.play_no = player_number
		self.total_dice = player_number * dice_number
		self.dice_number = dice_number
		total_players = []
		self.players=[]
		total_players.append(
			ComputerPlayer(
				name=name,
				dice_number=self.dice_number,
				game=self
			)
		)
		for i in range(0, player_number - 1):
			total_players.append(
				TotalRandom(
					name=i,
					dice_number=self.dice_number,
					game=self
				)
			)


		odds = []
		rounds = []
		round_number=0
		reward = 0

		for i in tqdm(range(3000)):

			for k in range(0, len(total_players)):
				total_players[k].reset()
				self.players.append(total_players[k])

			random.shuffle(self.players)
			self.first_player = random.choice(self.players)
			round_number += 1
			rounds.append(round_number)

			while len(self.players) > 1:
				self.run_round()
			self.players[0].score += 1
			self.players = []
			self.total_dice = player_number * dice_number
			
			for j in range(0, player_number):
				# self.players[j].score += 1
				if total_players[j].name == name:
					odds.append(total_players[j].score / round_number)
			
		plt.plot(rounds[10:], odds[10:])
		plt.xlabel('Iterations')
		plt.ylabel('Proportion of AI wins')
		plt.show()

	def run_round(self):
		self.round += 1
		for player in self.players:
			player.roll_dice()

		round_over = False
		current_bet = None
		current_player = self.first_player
		while not round_over:
			next_player = self.get_next_player(current_player)
			next_bet = current_player.make_bet(current_bet)
			
			if next_bet == DUDO:
				self.run_dudo(current_player, current_bet)
				round_over = True
			else:
				current_bet = next_bet

			if len(self.players) > 1:
				current_player = next_player


		def run_dudo(self, player, bet):
		previous_player = self.get_previous_player(player)
		dice_count = self.count_dice(bet.value)
		if dice_count < bet.quantity:
			self.first_player = player
			self.remove_die(previous_player)
		else:
			self.first_player = previous_player
			self.remove_die(player)
		self.total_dice=self.total_dice-1


	def count_dice(self, value):
		number = 0
		for player in self.players:
			number += player.count_dice(value)

		return number

	def remove_die(self, player):
		player.dice.pop()
		if len(player.dice) == 0:
			self.first_player = self.get_next_player(player)
			self.players.remove(player)
		elif len(player.dice) == 1 and player.palifico_round == -1:
			player.palifico_round = self.round + 1

	def is_palifico_round(self):
		if len(self.players) < 3:
			return False
		for player in self.players:
			if player.palifico_round == self.round:
				return True
		return False

	def get_random_name(self):
		random.shuffle(bot_names)
		return bot_names.pop()

	def get_next_player(self, player):
		return self.players[(self.players.index(player) + 1) % len(self.players)]

	def get_previous_player(self, player):
		return self.players[(self.players.index(player) - 1) % len(self.players)]

def get_argv(args, index, default):
	try:
		value = args[index]
	except IndexError:
		value = default
	return value

def main(args):
	name = get_argv(args, 1, 'Player')
	bot_number = int(get_argv(args, 2, 3))
	if bot_number < 1:
		print(INSUFFICIENT_BOTS)
		return
	dice_number = int(get_argv(args, 3, 4))
	if dice_number < 1:
		print(INSUFFICIENT_DICE)
		return

	Perudo(name, bot_number, dice_number)

if __name__ == '__main__':
	main(sys.argv)
