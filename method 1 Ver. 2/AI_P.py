import tensorflow as tf
from tensorflow import keras
from AI import AI
import random
import numpy as np
from bet import DUDO
from bet import create_bet
from bet_exceptions import BetException
from math import ceil

action_dim = 19
class PGModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(200, activation='relu')
        self.dense2 = keras.layers.Dense(200, activation='relu')
        keras.layers.Dropout(0.1)
        self.all_acts = keras.layers.Dense(units=action_dim)
        self.x = 0

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.all_acts(x)
        self.x = x
        output = tf.nn.softmax(x)
        return output


class PG():
    def __init__(self):
        self.model = PGModel()

    def choose_action(self, s):
        prob = self.model.predict(np.array([s]))[0]
        for i in range(len(prob)):
            if prob[i] != prob[i]:
                prob[i] = 0
        if np.sum(prob) != 0:
            prob = prob / np.sum(prob)
        else:
            prob = np.ones(19)
            prob = prob / np.sum(prob)
        return np.random.choice(len(prob), p=prob)

    def discount_reward(self, rewards, final_re, gamma=0.9):  # 衰减reward 通过最后一步奖励反推真实奖励
        out = np.zeros_like(rewards)
        dis_reward = final_re

        for i in reversed(range(len(rewards))):
            dis_reward = dis_reward + gamma * rewards[i]  # 前一步的reward等于后一步衰减reward加上即时奖励乘以衰减因子
            out[i] = dis_reward
        return out / np.std(out - np.mean(out))

    def all_actf(self):
        all_act = self.model.x
        return all_act

    def reca_batch(self, a_batch):
        a = a_batch
        return a

    def train(self, records, final_re):  # 训练
        s_batch = np.array([record[0] for record in records])# 取状态，每次batch个状态
        a_batch = np.array([[1 if record[1] == i else 0 for i in range(action_dim)]
                            for record in records])
        self.reca_batch(a_batch)
        prob_batch = self.model.predict(s_batch) * a_batch
        r_batch = self.discount_reward([record[2] for record in records], final_re)

        self.model.compile(loss=self.def_loss, optimizer=keras.optimizers.Adam(0.001))
        self.model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=1)

    def def_loss(self, label=reca_batch, logit=all_actf):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
        return neg_log_prob


class AI_P(AI):

    def __init__(self, name, dice_number, game):
        self.method = PG()
        self.record = []
        super(AI_P, self).__init__(name, dice_number, game)

    def make_bet(self, current_bet):
        if current_bet is None:
            total_dice_estimate = len(self.dice) * len(self.game.players)
            value = random.choice(self.dice).value
            quantity_limit = (total_dice_estimate - len(self.dice)) / 6

            if value > 1:
                quantity_limit *= 2

            quantity = self.count_dice(value) + random.randrange(0, ceil(quantity_limit + 1))
            bet = create_bet(quantity, value, current_bet, self, self.game)
        else:
            p = self.prob(self.game.total_dice, current_bet.quantity - self.count_dice(current_bet.value), current_bet.value)
            abservation = np.array([current_bet.value, current_bet.quantity, p, self.game.total_dice])
            action = self.method.choose_action(abservation)
            if action == 0:
                bet = DUDO
                reward = self.reward(current_bet) * 10 * (self.game.play_no - len(self.game.players))
            else:
                value = action % 6
                quantity = ceil(action / 6) + current_bet.quantity
                try:
                    bet = create_bet(quantity, value, current_bet, self, self.game)
                    reward = - self.reward(bet)
                except BetException:
                    bet = None
                    reward = -1
            self.record.append((abservation, action, reward))
        return bet

    def reset(self):
        if len(self.record) > 1:
            if np.sum([record[2] for record in self.record]) != 0:
                self.method.train(self.record, self.rewards)
                self.record = []
            else:
                pass
        else:
            pass
        super(AI_P, self).reset()



