import tensorflow
import tensorforce
from tensorforce.agents import PPOAgent, DQNAgent
import pydealer
import numpy as np
new_ranks = {
    "values": {
        "Ace": 11,
        "King": 10,
        "Queen": 10,
        "Jack": 10,
        "10": 10,
        "9": 9,
        "8": 8,
        "7": 7,
        "6": 6,
        "5": 5,
        "4": 4,
        "3": 3,
        "2": 2
    }
}
class gameSim():
    player = []
    dealer = []
    deck = pydealer.Deck(rebuild=True, re_shuffle=True)
    def __init__(self):
        self.deck.shuffle()
        self.player = [new_ranks['values'][self.deck.deal(1)[0].value], new_ranks['values'][self.deck.deal(1)[0].value]]
        self.dealer = [new_ranks['values'][self.deck.deal(1)[0].value], new_ranks['values'][self.deck.deal(1)[0].value]]

    def new_hand(self):
        self.deck = pydealer.Deck(ranks=new_ranks)
        self.deck.shuffle()
        self.player = [new_ranks['values'][self.deck.deal(1)[0].value], new_ranks['values'][self.deck.deal(1)[0].value]]
        self.dealer = [new_ranks['values'][self.deck.deal(1)[0].value], new_ranks['values'][self.deck.deal(1)[0].value]]
    
    def hit(self, player):
        if player:
            self.player.append(new_ranks['values'][self.deck.deal(1)[0].value])
        else:
            self.dealer.append(new_ranks['values'][self.deck.deal(1)[0].value])

    def value(self, player):
        if player:
            return sum(self.player)
        else:
            return sum(self.dealer)

    def bust(self, player):
        return self.value(player) > 21

    def win(self, player):
        return self.value(player) > self.value(not player)

    def run_dealer(self):
        while not self.bust(False) and not self.win(False):
            self.hit(False)
    
    def state(self):
        ret = [0]*10
        ret[0:len(self.player)] = self.player
        ret[-1] = self.dealer[0]
        return [float(i) for i in ret]

    def reward(self):
        if self.bust(True):
            return -1
        elif self.bust(False):
            return 1
        elif self.win(False):
            return -1
        else:
            return 1
