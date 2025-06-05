import numpy as np
from matplotlib import pyplot as plt

target_fpath = '/flare/hpe_dragon_collab/balin/archit_fine_tune/smile_regress.training.log'
case_files = [
    #'training_avdata_regNtransDrop.log',
    #'training_avdata_regDrop.log',
    #'training_avdata_noDrop.log',
    'training_avdata_regDrop_CR.log',
    #'training_wfdata_regDrop_CR_random.log',
    'training_wfdata_regDrop_CR_stratified.log',
    'training_wfdata_regDrop_CR_stratified_realFT_2-6.log',
    'training_wfFulldata_regDrop_CR_stratified_realFT_2-6.log',
    'training_wfFulldata_regDrop_CR_stratified.log',
    'training_wfFulldata_regDrop_CR_stratified_realFT_3-6.log'
]
labels = ['loss','mae','r2','val_loss','val_mae','val_r2']

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

cases = {}
for case in case_files:
    cases[case] = {}
    for label in labels:
        cases[case][label] = []
    with open(case,'r') as fh:
        for l in fh:
            if "loss:" in l:
                parsed_l = l.split(' ')
                for i in range(len(parsed_l)):
                    tmp = parsed_l[i].replace(':','')
                    if tmp in labels:
                        cases[case][tmp].append(float(parsed_l[i+1]))

target = {}
with open(target_fpath,'r') as fh:
    headers = fh.readline().split(',')
    for header in headers:
        target[header.rstrip('\n')] = []
    for l in fh:
        data = l.split(',')
        for i, key in enumerate(target):
            target[key].append(float(data[i]))


# Plot Train R2
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
axs.plot(target['r2'],label = "Archit's fine-tune",ls="-",linewidth=2,color='black')
for case in cases.keys():
    axs.plot(cases[case]['r2'],label=case,ls="--",linewidth=2)
#axs.set_yscale("log")
axs.grid()
axs.set_xlabel('Training Epochs')
axs.set_ylabel('Train R2')
axs.legend(loc='lower right')
fig.savefig('plt_train_r2.png')

# Plot Val R2
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
axs.plot(target['val_r2'],label = "Archit's fine-tune",ls="-",linewidth=2,color='black')
for case in cases.keys():
    axs.plot(cases[case]['val_r2'],label=case,ls="--",linewidth=2)
#axs.set_yscale("log")
axs.grid()
axs.set_xlabel('Training Epochs')
axs.set_ylabel('Validation R2')
axs.legend(loc='lower right')
fig.savefig('plt_val_r2.png')