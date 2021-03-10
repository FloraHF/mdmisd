from game_2dsir0 import TDSISDPointCapGame
from base_2dsir0 import strategy_barrier, strategy_pass, strategy_default



# first set of strategy
g = TDSISDPointCapGame(1, 1.2)
x1, x2, xi = g.play(dstr=strategy_barrier, 
					istr=strategy_barrier, render=True)