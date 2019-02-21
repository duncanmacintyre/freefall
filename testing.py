# testing.py
# use for testing the functions

import numpy
import pandas
import matplotlib.pyplot as plt
import freefall
from time import sleep

df = freefall.box_sim(l=6, h=0.2, w=0.4, m=10, W_x=4, W_y=0.3, W_z=0.1, start=0, stop=10, dt=0.0001)

fig, ax = freefall.plot(df, 0, 10, 'The first ten seconds', sharex=True)
plt.show()

# import vpython
# sleep(7)
# freefall.visualize_df(df, 1.8, 4, 0.4, speed=0.04, stop=2)