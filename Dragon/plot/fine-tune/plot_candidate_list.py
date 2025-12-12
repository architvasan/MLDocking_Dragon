import numpy as np
import glob
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
from matplotlib import pyplot as plt
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

class Candidates():
    def __init__(self, path):
        self.path = path
        self.stats = {
            'smiles': [],
            'scores': [],
            'x': None,
            'y': None
        }

        self.parse_file()
        self.compute_kde()

    def parse_line(self, l):
        parsed_l = l.split(" ")
        parsed_l = [item for item in parsed_l if item]
        smiles = parsed_l[0]
        score = float(parsed_l[1]) * -1.
        return smiles, score

    def parse_file(self):
        with open(self.path,'r') as fh:
            print(f"Reading file: {self.path}")
            for l in fh:
                if "smiles" in l:
                    continue
                else:
                    smiles, score = self.parse_line(l)
                    self.stats["smiles"].append(smiles)
                    self.stats["scores"].append(score)

    def compute_kde(self):
        kde = gaussian_kde(self.stats['scores'])
        self.stats['x'] = np.linspace(min(self.stats['scores']), max(self.stats['scores']), 1000)
        self.stats['y'] = kde(self.stats['x'])

class Simulated():
    def __init__(self,path):
        self.path = path
        self.data = {
            'smiles': [],
            'scores': []
        }

        self.parse_files()

    def parse_line(self, l):
        parsed_l = l.split(" ")
        parsed_l = [item for item in parsed_l if item]
        smiles = parsed_l[0]
        score = float(parsed_l[1]) * -1.
        return smiles, score

    def parse_files(self):
        files = glob.glob(self.path+"/simulated_compounds*")
        for file in files:
            print('Reading file: ', file)
            with open(file,'r') as fh:
                for l in fh:
                    if "smiles" in l:
                        continue
                    else:
                        smiles, score = self.parse_line(l)
                        if smiles not in self.data["smiles"]:
                            self.data["smiles"].append(smiles)
                            self.data["scores"].append(score)

    def compute_accuracy(self,pred_smiles,pred_scores):
        sim_values = []
        for i in range(len(pred_smiles)):
            found = False
            for j in range(len(self.data["smiles"])):
                if pred_smiles[i] == self.data["smiles"][j]:
                    sim_values.append(self.data["scores"][j])
                    found = True
                    break
            if not found:
                pred_scores[i] = -99
        pred_scores = [i for i in pred_scores if i > -90]
        #print(f'Computing R2 score with {len(pred_scores)} values')
        return r2_score(sim_values, pred_scores), mean_absolute_error(sim_values, pred_scores), len(pred_scores)

system = "aurora"
if system == "aurora":
    root = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25'
elif system == "local":
    root = '/Users/riccardobalin/Documents/ALCF/Conferences/PASC25'

i0 = Candidates(root+"/runs/fine_tune/try_3/top_candidates_0.out")
i1 = Candidates(root+"/runs/fine_tune/try_3/top_candidates_1.out")
i2 = Candidates(root+"/runs/fine_tune/try_3/top_candidates_2.out")
i3 = Candidates(root+"/runs/fine_tune/try_3/top_candidates_3.out")

# Plot score distribution
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
#axs.plot(i0.stats['x'],i0.stats['y'],label="iter 1",ls="-",linewidth=2)
#axs.hist(i0.stats['scores'],bins=10, edgecolor='black',density=True)
#axs.plot(i1.stats['x'],i1.stats['y'],label="iter 2",ls="-",linewidth=2)
#axs.plot(i2.stats['x'],i2.stats['y'],label="iter 3",ls="-",linewidth=2)
#axs.plot(i3.stats['x'],i3.stats['y'],label="iter 4",ls="-",linewidth=2)
sns.kdeplot(i0.stats["scores"],fill=False,label="iter 1",ls="-",linewidth=2)
sns.kdeplot(i1.stats["scores"],fill=False,label="iter 2",ls="-",linewidth=2)
sns.kdeplot(i2.stats["scores"],fill=False,label="iter 3",ls="-",linewidth=2)
sns.kdeplot(i3.stats["scores"],fill=False,label="iter 4",ls="-",linewidth=2)

axs.grid()
axs.set_xlabel('Docking Score')
axs.set_ylabel('Density')
axs.set_title('Docking Score PDF of Top Candidates')
axs.legend(loc='upper left')
fig.savefig('plt_scores.png')


sim = Simulated(root+"/runs/fine_tune/try_3")
i0_r2, i0_mae, i0_n = sim.compute_accuracy(i0.stats["smiles"],i0.stats["scores"])
print(f'Iter 0 with {i0_n} values: R2={i0_r2}, MAE={i0_mae}')
i1_r2, i1_mae, i1_n = sim.compute_accuracy(i1.stats["smiles"],i1.stats["scores"])
print(f'Iter 1 with {i1_n} values: R2={i1_r2}, MAE={i1_mae}')
i2_r2, i2_mae, i2_n = sim.compute_accuracy(i2.stats["smiles"],i2.stats["scores"])
print(f'Iter 2 with {i2_n} values: R2={i2_r2}, MAE={i2_mae}')
i3_r2, i3_mae, i3_n = sim.compute_accuracy(i3.stats["smiles"],i3.stats["scores"])
print(f'Iter 3 with {i3_n} values: R2={i3_r2}, MAE={i3_mae}')

