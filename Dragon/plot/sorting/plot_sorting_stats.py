import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)


class SortScaling:
    def __init__(self,path,case_list):
        self.base_path = path
        self.case_list = case_list
        self.n_cases = len(case_list)
        self.stats = {
            'sort_time': np.zeros((self.n_cases,)),
            'sort_ddict_time': np.zeros((self.n_cases,)),
        }
        self.counts = {
            'sort_time': np.zeros((self.n_cases,)),
            'sort_ddict_time': np.zeros((self.n_cases,)),
        }

    def avg(self, my_list):
        return sum(my_list)/len(my_list)

    def parse_files(self):
        for i in range(self.n_cases):
            path = self.base_path+f"/{self.case_list[i]}/mldocking*"
            # Loop over runs found
            run_files = glob.glob(path)
            for run_file in run_files:
                print('Reading file: ', run_file)
                with open(run_file,'r') as fh:
                    ddict_keys = []
                    ddict_mem = []
                    for l in fh:
                        if "Performed sorting of 10000 compounds" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['sort_time'][i] += float(l.split(':')[-1].split(',')[0].split('=')[-1])
                            self.counts['sort_time'][i] += 1
                            self.stats['sort_ddict_time'][i] += float(l.split(':')[-1].split(',')[-1].split('=')[-1])
                            self.counts['sort_ddict_time'][i] += 1

        # Divide by the counts for each gpu number to get the average over runs
        for key in self.stats.keys():
            val = self.stats[key]
            if len(val.shape)==1:
                #self.train_fom[key] = np.divide(val, self.counts[key], where=self.counts[key]>0)
                np.divide(val, self.counts[key], out=self.stats[key], where=self.counts[key]>0)
            else:
                for j in range(val.shape[1]):
                    #self.train_fom[key][:,j] = np.divide(val[:,j], self.counts[key], where=self.counts[key]>0)
                    np.divide(val[:,j], self.counts[key], out=self.stats[key][:,j], where=self.counts[key]>0)


base_path = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25/runs/sorting/131072'
case_list = ['32_1m','32_2m','32_4m']
mans = SortScaling(base_path,case_list)
mans.parse_files()

base_path = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25/runs/sorting/131072'
case_list = ['16_2m','32_2m','64_2m']
nds = SortScaling(base_path,case_list)
nds.parse_files()


###############################
# Plot data loader IO and DDict times vs. manager
managers = [1,2,4]
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.plot(managers, mans.stats['sort_time'],label = "Run time",marker="o",ls="-",markersize=10, linewidth=2)
ideal = [mans.stats['sort_time'][0]/(man/1) for man in managers]
axs.plot(managers, ideal,ls="--", linewidth=1, color="k")
efficiency = [id/real*100 for id,real in zip(ideal,mans.stats['sort_time'].tolist())]
for i in range(len(efficiency)):
    axs.text(managers[i] + 0.1, mans.stats['sort_time'][i]+ 0.1, str(round(efficiency[i],1))+" %", fontsize=9, color='black')
ddict_frac = mans.stats['sort_ddict_time']/mans.stats['sort_time']*100
print(ddict_frac)

axs.grid()
axs.set_xlabel('Number of DDict Managers')
#fig.legend(bbox_to_anchor=(1.25,0.7))
axs.legend(loc='upper right')
axs.set_ylabel('Time [sec]')
axs.set_title('Sorting Run Time')
#axs.set_yscale("log")
fig.tight_layout(pad=2.0)
fig.savefig("plt_sort_mans.png")

###############################
# Plot data loader IO and DDict times vs. manager
nodes = [16,32,64]
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.plot(nodes, nds.stats['sort_time'],label = "Run time",marker="o",ls="-",markersize=10, linewidth=2)
ideal = [nds.stats['sort_time'][0]/(nd/nodes[0]) for nd in nodes]
axs.plot(nodes, ideal,ls="--", linewidth=1, color="k")
efficiency = [id/real*100 for id,real in zip(ideal,nds.stats['sort_time'].tolist())]
for i in range(len(efficiency)):
    axs.text(nodes[i] + 0.1, nds.stats['sort_time'][i]+ 0.1, str(round(efficiency[i],1))+" %", fontsize=9, color='black')
ddict_frac = mans.stats['sort_ddict_time']/mans.stats['sort_time']*100
print(ddict_frac)

axs.grid()
axs.set_xlabel('Number of Nodes')
#fig.legend(bbox_to_anchor=(1.25,0.7))
axs.legend(loc='upper right')
axs.set_ylabel('Time [sec]')
axs.set_title('Sorting Run Time')
#axs.set_yscale("log")
fig.tight_layout(pad=2.0)
fig.savefig("plt_sort_nodes.png")
