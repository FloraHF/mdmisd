from game_2dsir0 import TDSISDPointCapGame
from game_2dsir0 import strategy_barrier, strategy_pass



# first set of strategy
g = TDSISDPointCapGame(1, 1.2)
x1, x2, xi = g.play(dstr=strategy_pass, 
					istr=strategy_barrier, render=True)