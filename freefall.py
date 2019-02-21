# freefall.py
# collection of functions for modelling the rotational motion of falling objects

import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from time import sleep

# this function will return the similar angle in [0, 2 pi) for a given angle
def correct_angle(n):
	while n < 0:
		n += 2*numpy.pi
	if n > 2*numpy.pi:
		n = n % (2*numpy.pi)
	return(n)

# this generator function yields the values of all variables for each Euler step in the freefall rotation simulation
def euler_step(I_x, I_y, I_z, x, y, z, W_x, W_y, W_z, start, stop, dt):

	# set up variables
	I_x = numpy.float(I_x) # moments of inertia
	I_y = numpy.float(I_y)
	I_z = numpy.float(I_z)
	x = numpy.float(x) # angular positions
	y = numpy.float(y)
	z = numpy.float(z)
	W_x = numpy.float(W_x) # angular velocities
	W_y = numpy.float(W_y)
	W_z = numpy.float(W_z)
	a_x = numpy.float() # angular accelerations
	a_y = numpy.float()
	a_z = numpy.float()
	a_x = numpy.float() # old angular accelerations (for n-1; used to calculate angular jerk)
	a_y = numpy.float()
	a_z = numpy.float()
	j_x = numpy.float() # angular jerk
	j_y = numpy.float()
	j_z = numpy.float()
	t = numpy.float(start) # time
	stop = numpy.float(stop) # stop time
	dt = numpy.float(dt) # time step
	
	# main loop function
	for n in range(int((stop - start) / dt) + 1):

		# store the old angular accelerations
		a_x_old = a_x
		a_y_old = a_y
		a_z_old = a_z
		
		# calculate angular accelerations from angular velocities, using Euler's equations
		a_x = (I_y - I_z) * W_y * W_z / I_x
		a_y = (I_z - I_x) * W_z * W_x / I_y
		a_z = (I_x - I_y) * W_x * W_y / I_z
		
		# calculate and store angular jerk (derivative of angular acceleration)
		if n > 0: # no jerk for first data point
			j_x = (a_x - a_x_old) / dt
			j_y = (a_y - a_y_old) / dt
			j_z = (a_z - a_z_old) / dt

		# yield result
		yield(t, x, y, z, W_x, W_y, W_z, a_x, a_y, a_z, j_x, j_y, j_z)

		# calculate angular velocities from angular accelerations
		W_x += a_x * dt
		W_y += a_y * dt
		W_z += a_z * dt

		# calculate angular positions from angular velocities
		x = correct_angle(x + W_x * dt)
		y = correct_angle(y + W_y * dt)
		z = correct_angle(z + W_z * dt)

		t += dt

# this function does a complete Euler method run based on moments of inertia, returning a pandas.DataFrame of the results
def I_sim(I_x, I_y, I_z, x, y, z, W_x, W_y, W_z, start, stop, dt):

	# return a pandas.DataFrame constructed from the euler_step() generator function
	return(pandas.DataFrame(euler_step(I_x=I_x, I_y=I_y, I_z=I_z, x=x, y=y, z=z, W_x=W_x, W_y=W_y, W_z=W_z, start=start, stop=stop, dt=dt), columns=('t', 'x', 'y', 'z', 'W_x', 'W_y', 'W_z','a_x', 'a_y', 'a_z', 'j_x', 'j_y', 'j_z')))

# this function runs the box simulation, returning a pandas.DataFrame
def box_sim(l, h, w, m, W_x, W_y, W_z, start, stop, dt):

	# calculate principle moments of inertia
	I_x = (1.0/12.0) * m * (w**2 + h**2) # length axiss
	I_y = (1.0/12.0) * m * (l**2 + w**2) # height axis
	I_z = (1.0/12.0) * m * (l**2 + h**2) # width axis

	# string description of initial parameters
	descriptor = 'Moments of intertia: ({}, {}, {})\nInitial angular velocities: ({}, {}, {})'.format(*(round(j, 3) for j in (I_x, I_y, I_z, W_x, W_y, W_z, 0)))
	print(descriptor) # print this description
	
	# construct a pandas.DataFrame using I_sim() and return it
	return(I_sim(I_x=I_x, I_y=I_y, I_z=I_z, x=0, y=0, z=0, W_x=W_x, W_y=W_y, W_z=W_z, start=start, stop=stop, dt=dt))

# this function will plot the angular position, angular velocity, angular acceleration, and angular jerk for a given DataFrame over a given domain
def plot(df, start=None, stop=None, title=None, sharex=False):

	# font preferences for axes titles
	font = {'fontsize':13, 'fontname':'Source Sans Pro', 'fontweight':'medium'}
	
	if start is not None and stop is not None:
		df = df[(df['t'] >= start) & (df['t'] <= stop)]
	
	# set up the figure with four subplots
	figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(7, 10), sharex=sharex)

	# use the first subplot for angular position
	ax1.plot(df['t'], df['x'], 'bo', markersize=0.2, label='x')
	ax1.plot(df['t'], df['y'], 'ro', markersize=0.2, label='y')
	ax1.plot(df['t'], df['z'], 'go', markersize=0.2, label='z')
	ax1.grid(b=True, which='major', alpha=1) # major grid
	ax1.grid(b=True, which='minor', alpha=0.2) # minor grid
	ax1.set_xlim(start, stop) # set domain
	ax1.xaxis.set_minor_locator(AutoMinorLocator(10)) # show some minor ticks
	ax1.set_ylabel('Angle', **font)
	ax1.legend()
	
	# use the second subplot for angular velocity
	ax2.plot(df['t'], df['W_x'], 'b-', label='W_x')
	ax2.plot(df['t'], df['W_y'], 'r-', label='W_y')
	ax2.plot(df['t'], df['W_z'], 'g-', label='W_z')
	ax2.grid(b=True, which='major', alpha=1) # major grid
	ax2.grid(b=True, which='minor', alpha=0.2) # minor grid
	ax2.set_xlim(start, stop) # set domain
	ax2.xaxis.set_minor_locator(AutoMinorLocator(10)) # show some minor ticks
	ax2.set_ylabel('Angular velocity', **font)
	ax2.legend()
	
	# use the third subplot for angular acceleraiton
	ax3.plot(df['t'], df['a_x'], 'b-', label='W_x')
	ax3.plot(df['t'], df['a_y'], 'r-', label='W_y')
	ax3.plot(df['t'], df['a_z'], 'g-', label='W_z')
	ax3.grid(b=True, which='major', alpha=1) # major grid
	ax3.grid(b=True, which='minor', alpha=0.2) # minor grid
	ax3.set_xlim(start, stop) # set domain
	ax3.xaxis.set_minor_locator(AutoMinorLocator(10)) # show some minor ticks
	ax3.set_ylabel('Angular acceleration', **font)
	ax3.legend()
	
	# use the fourth subplot for angular jerk
	ax4.plot(df['t'], df['j_x'], 'b-', label='j_x')
	ax4.plot(df['t'], df['j_y'], 'r-', label='j_y')
	ax4.plot(df['t'], df['j_z'], 'g-', label='j_z')
	ax4.grid(b=True, which='major', alpha=1) # major grid
	ax4.grid(b=True, which='minor', alpha=0.2) # minor grid
	ax4.set_xlim(start, stop) # set domain
	ax4.xaxis.set_minor_locator(AutoMinorLocator(10)) # show some minor ticks
	ax4.set_ylabel('Angular jerk', **font)
	ax4.set_xlabel('Time (s)', **font)
	ax4.legend()
	
	# if a title has been set, display the title and align
	if title is not None:
		figure.suptitle(title, fontsize=16, fontname=font['fontname'], fontweight=font['fontweight'])
		figure.tight_layout(rect=[0, 0, 1, 0.95])
	# alignment for no title
	else:
		figure.tight_layout()
	
	# return the figure and axes
	return(figure, (ax1, ax2, ax3, ax4))

# this function runs a vpython box visualization based on an iterator (data) which yields four values each step in the form (t, x, y, z) as the first four values
def visualize(data, l, h, w, speed=1, caption=''):
	
	import vpython
	from time import sleep
	
	# set up the box object
	box = vpython.box(pos=vpython.vector(0, 0, 0), size=vpython.vector(l, h, w), color=vpython.color.red) # initialize as box object

	# main loop
	t_old, x_old, y_old, z_old = 0, 0, 0, 0 # these variables will hold the respective values from the previous Euler step
	is_first_step = True
	for t, x, y, z, *extra in data:

		# wait dt
		# it is important not to sleep the first time as we do not know what t starts at
		if is_first_step: # if first step
			is_first_step = False # set is_first_step to false
		else: # if not first step
			sleep((t - t_old) / speed) # wait dt, adjusted for the speed factor

		box.rotate(x - x_old, axis=vpython.vector(1,0,0)) # x-rotation
		box.rotate(y - y_old, axis=vpython.vector(0,1,0)) # y-rotation
		box.rotate(z - z_old, axis=vpython.vector(0,0,1)) # z-rotation

		# update the caption
		vpython.scene.caption = '\nt = {}\n{}'.format(round(t, 3), caption)

		# save variable values
		t_old, x_old, y_old, z_old = t, x, y, z

# this function will run the vpython box visualization for a pandas.DataFrame of simulation data
def visualize_df(df, l, h, w, speed=1, caption='', start=None, stop=None):

	# slice df to correct size based on start and stop
	if start is not None and stop is not None: # if domain fully specified
		df = df[(df['t'] >= start) & (df['t'] <= stop)] # slice based on t
	elif start is not None: # if only start point is specified
		df = df[df['t'] >= start] # slice based on t
	elif stop is not None: # if only end point is specified
		df = df[df['t'] <= stop] # slice based on t

	# use zip() to generate iterator (this should be faster than just iterating through the pandas.DataFrame)
	visualize(data=zip(df['t'], df['x'], df['y'], df['z']), l=l, h=h, w=w, speed=speed, caption=caption)
