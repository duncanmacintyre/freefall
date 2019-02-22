import numpy
import pandas
import matplotlib.pyplot as plt
import freefall

# generate the simulation data
df, I = freefall.box_sim(l=1.8, h=4, w=0.4, m=10, W_x=4, W_y=0.3, W_z=0.1, start=0, stop=10, dt=0.0001)

# plotting
fig, ax = freefall.plot(df, title='The first ten seconds')
plt.show()

# show a visualization
freefall.visualize_df(df, 1.8, 4, 0.4, speed=1, caption='This is the caption.')