import numpy
import pandas
import os
import datetime
import freefall

# SETUP
I = (4, 3, 2) # the moment of inertia components to use
A, B, C = 5, 5, 5 # the final values for the indeces of iteration
directory = 'runW12' # the directory in which to save data and plots
# this numpy.array will hold metadata about the runs
run_log = numpy.empty(shape=(A*B*C), dtype=[('run','uint16'), ('filename','S13'), ('W_x','uint8'), ('W_y','uint8'), ('W_z','uint8'), ('n_x','uint16'), ('n_y','uint16'), ('n_z','uint16')])
i = 0 # this index will incriment by 1 for each Euler method run
# make the directories for storing data (if any already exist, an error will be raised)
os.makedirs('{}/data'.format(directory))
os.makedirs('{}/SVG_fingerprint'.format(directory))
os.makedirs('{}/PNG_fingerprint'.format(directory))
os.makedirs('{}/SVG_10s'.format(directory))
os.makedirs('{}/PNG_10s'.format(directory))
os.makedirs('{}/SVG_50s'.format(directory))
os.makedirs('{}/PNG_50s'.format(directory))
# print end-of-setup notification
print('{}\nFinished setup. Beginning simulations.'.format(directory))

# ITERATION
for a in range(A): # index controlling W_x
	for b in range(B): # index controlling W_y
		for c in range(C): # index controlling W_z

			# SETUP
			# calculate the initial angular velocities for this run
			W = (a, b, c) # angular velocities
			
			# EULER METHOD RUN
			# run the simulation
			df = freefall.I_sim(I_x=I[0], I_y=I[1], I_z=I[2], x=0, y=0, z=0, W_x=W[0], W_y=W[1], W_z=W[2], start=0, stop=50, dt=0.0001, store_every=200)
			# get the final n-values
			n = (df['n_x'].iloc[-1], df['n_y'].iloc[-1], df['n_z'].iloc[-1])

			# SAVING
			# generate the filename for saving
			filename = '{:04d}_{:02d}_{:02d}_{:02d}'.format(i, *W) # the filename will include the run number and angular velocity components
			# log the run number, filename, initial velocities, and final n-values
			run_log[i] = (i, filename, *W, *n)
			# save the simulation data
			df.to_csv('{}/data/{}.csv'.format(directory, filename))

			# PLOTTING
			fig1, ax1 = freefall.plot(df, 0, 10, 'W = ({:.1f}, {:.1f}, {:.1f})'.format(*W, n), sharex=False) # plot of first ten seconds
			fig2, ax2 = freefall.plot(df, 0, 50, 'W = ({:.1f}, {:.1f}, {:.1f})'.format(*W, n), sharex=False) # plot of all 100 seconds
			# save figures as SVGs and PNGs
			fig1.savefig('{}/SVG_10s/{}.svg'.format(directory, filename))
			fig1.savefig('{}/PNG_10s/{}.png'.format(directory, filename))
			fig2.savefig('{}/SVG_50s/{}.svg'.format(directory, filename))
			fig2.savefig('{}/PNG_50s/{}.png'.format(directory, filename))

			# PLOTTING OF FINGERPRINT GRAPHS
			figF, axF = freefall.fingerprint(df, title='W = ({:.1f}, {:.1f}, {:.1f})'.format(*W, n)) # plot of first ten seconds
			# save figures as SVGs and PNGs
			figF.savefig('{}/SVG_fingerprint/{}.svg'.format(directory, filename))
			figF.savefig('{}/PNG_fingerprint/{}.png'.format(directory, filename))

			# print end of run notification
			print('Finished run {:04d} for W = ({:.1f}, {:.1f}, {:.1f}) ({} of {} complete).'.format(i, *W, i+1, A*B*C))

			# incriment index
			i += 1

# SAVE LOGS
# save a description of run metadata
with open('{}/about.txt'.format(directory), 'w') as aboutfile:
    aboutfile.write('Run {}\nI = ({}, {}, {})\nFinished {}'.format(directory, *I, datetime.datetime.now()))
# use pandas to save the log to a CSV
pandas.DataFrame(run_log, columns=('run', 'filename', 'W_x', 'W_y', 'W_z', 'n_x', 'n_y', 'n_z')).to_csv('{}/run_log.csv'.format(directory), index=False)
# print final notification
print('{} complete'.format(directory))
