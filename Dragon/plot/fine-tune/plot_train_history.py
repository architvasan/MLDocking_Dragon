import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

class History():
    def __init__(self,path):
        self.path = path
        self.labels = ['loss','mae','r2','val_loss','val_mae','val_r2']
        self.stats = {}
        for label in self.labels:
            self.stats[label] = []

        self.parse_file()
    
    def parse_history_line(l):
        vals = []
        parsed_l = l.split(' ')
        for i in range(len(parsed_l)):
            if parsed_l[i] == 'mse:':
                vals.append(float(parsed_l[i+1]))
            if parsed_l[i] == 'r2:':
                vals.append(float(parsed_l[i+1]))
            if parsed_l[i] == 'val_mse:':
                vals.append(float(parsed_l[i+1]))
            if parsed_l[i] == 'val_r2:':
                vals.append(float(parsed_l[i+1]))
        return vals

    def parse_file(self):
        with open(self.path,'r') as fh:
            for l in fh:
                if "loss:" in l:
                    parsed_l = l.split(' ')
                    for i in range(len(parsed_l)):
                        tmp = parsed_l[i].replace(':','')
                        if tmp in self.labels:
                            self.stats[tmp].append(float(parsed_l[i+1]))


system = "aurora"
if system == "aurora":
    root = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25'
elif system == "local":
    root = '/Users/riccardobalin/Documents/ALCF/Conferences/PASC25'

i0 = History(root+"/runs/fine_tune/try_4/training_0.log")
i1 = History(root+"/runs/fine_tune/try_4/training_1.log")
i2 = History(root+"/runs/fine_tune/try_4/training_2.log")
i3 = History(root+"/runs/fine_tune/try_4/training_3.log")

# Plot Train R2
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
axs.plot(i0.stats['r2'],label="iter 1",ls="-",linewidth=2)
axs.plot(i1.stats['r2'],label="iter 2",ls="-",linewidth=2)
axs.plot(i2.stats['r2'],label="iter 3",ls="-",linewidth=2)
axs.plot(i3.stats['r2'],label="iter 4",ls="-",linewidth=2)
#axs.set_yscale("log")
axs.grid()
axs.set_xlabel('Training Epochs')
axs.set_ylabel('R2 (Coeff. of Determination)')
axs.set_title('Training Set')
axs.legend(loc='lower right')
fig.savefig('plt_train_r2.png',bbox_inches='tight')

# Plot Val R2
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
axs.plot(i0.stats['val_r2'],label="iter 1",ls="-",linewidth=2)
axs.plot(i1.stats['val_r2'],label="iter 2",ls="-",linewidth=2)
axs.plot(i2.stats['val_r2'],label="iter 3",ls="-",linewidth=2)
axs.plot(i3.stats['val_r2'],label="iter 4",ls="-",linewidth=2)
#axs.set_yscale("log")
axs.grid()
axs.set_xlabel('Training Epochs')
axs.set_ylabel('R2 (Coeff. of Determination)')
axs.set_title('Validation Set')
axs.legend(loc='lower right')
axs.set_ylim(0,1)
fig.savefig('plt_val_r2.png',bbox_inches='tight')