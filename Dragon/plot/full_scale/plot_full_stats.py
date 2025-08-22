import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)


class WFStats:
    def __init__(self,path,cases,nodes):
        self.base_path = path
        self.cases = cases
        self.n_nodes = len(nodes)
        self.nodes = nodes
        self.stats = {
            'inf_ddict_init': np.zeros((self.n_nodes,)),
            'model_ddict_init': np.zeros((self.n_nodes,)),
            'pool_procs': np.zeros((self.n_nodes,)),
            'num_files': np.zeros((self.n_nodes,)),
            'load_time': np.zeros((self.n_nodes,)),
            'load_IO_avg_time': np.zeros((self.n_nodes,)),
            'load_IO_max_time': np.zeros((self.n_nodes,)),
            'load_ddict_avg_time': np.zeros((self.n_nodes,)),
            'load_ddict_max_time': np.zeros((self.n_nodes,)),
            'inf_io_time': np.zeros((self.n_nodes,)),
            'inf_time': np.zeros((self.n_nodes,)),
            'sort_time': np.zeros((self.n_nodes,)),
            'sort_io_time': np.zeros((self.n_nodes,)),
            'sim_time': np.zeros((self.n_nodes,)),
            'sim_io_time': np.zeros((self.n_nodes,)),
            'train_time': np.zeros((self.n_nodes,)),
            'train_io_time': np.zeros((self.n_nodes,)),
        }
        self.counts = {
            'inf_ddict_init': np.zeros((self.n_nodes,)),
            'model_ddict_init': np.zeros((self.n_nodes,)),
            'pool_procs': np.zeros((self.n_nodes,)),
            'num_files': np.zeros((self.n_nodes,)),
            'load_time': np.zeros((self.n_nodes,)),
            'load_IO_avg_time': np.zeros((self.n_nodes,)),
            'load_IO_max_time': np.zeros((self.n_nodes,)),
            'load_ddict_avg_time': np.zeros((self.n_nodes,)),
            'load_ddict_max_time': np.zeros((self.n_nodes,)),
            'inf_io_time': np.zeros((self.n_nodes,)),
            'inf_time': np.zeros((self.n_nodes,)),
            'sort_time': np.zeros((self.n_nodes,)),
            'sort_io_time': np.zeros((self.n_nodes,)),
            'sim_time': np.zeros((self.n_nodes,)),
            'sim_io_time': np.zeros((self.n_nodes,)),
            'train_time': np.zeros((self.n_nodes,)),
            'train_io_time': np.zeros((self.n_nodes,)),
        }

    def avg(self, my_list):
        return sum(my_list)/len(my_list)

    def parse_files(self):
        for i in range(self.n_nodes):
            if "_" in self.cases[i]:
                path = self.base_path+f"/{self.cases[i]}/ddict/mldocking*"
            else:
                path = self.base_path+f"/{self.cases[i]}/mldocking*"
            # Loop over runs found
            run_files = glob.glob(path)
            for run_file in run_files:
                inf_io_times = []
                print('Reading file: ', run_file)
                with open(run_file,'r') as fh:
                    for l in fh:
                        if "Launched Dragon Dictionary for inference" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['inf_ddict_init'][i] += float(l.split(' ')[-2])
                            self.counts['inf_ddict_init'][i] += 1
                        if "Launched Dragon Dictionary for model" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['model_ddict_init'][i] += float(l.split(' ')[-2])
                            self.counts['model_ddict_init'][i] += 1
                        if "Number of files to read is" in l:
                            self.stats['num_files'][i] += int(l.split(' ')[-1])
                            self.counts['num_files'][i] += 1
                        if "Number of Pool processes" in l:
                            self.stats['pool_procs'][i] += int(l.split(' ')[5])
                            self.counts['pool_procs'][i] += 1
                        if "Loaded inference data in" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['load_time'][i] += float(l.split(' ')[4])
                            self.counts['load_time'][i] += 1
                        if "IO times:" in l:
                            parsed_l=l.split(":")[-1].split(",")
                            self.stats['load_IO_avg_time'][i] += float(parsed_l[0].split("=")[-1].split(" ")[0])
                            self.counts['load_IO_avg_time'][i] += 1
                            self.stats['load_IO_max_time'][i] += float(parsed_l[1].split("=")[-1].split(" ")[0])
                            self.counts['load_IO_max_time'][i] += 1
                        if "DDict times:" in l:
                            parsed_l=l.split(":")[-1].split(",")
                            self.stats['load_ddict_avg_time'][i] += float(parsed_l[0].split("=")[-1].split(" ")[0])
                            self.counts['load_ddict_avg_time'][i] += 1
                            self.stats['load_ddict_max_time'][i] += float(parsed_l[1].split("=")[-1].split(" ")[0])
                            self.counts['load_ddict_max_time'][i] += 1
                        if "Performed inference on" in l:
                            parsed_l = l.split(":")[-1]
                            ddict_l = parsed_l.split(",")[1]
                            inf_io_times.append(float(ddict_l.split("=")[-1]))
                        if "Performed inference in" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            print(float(l.split(" ")[-2]))
                            self.stats["inf_time"][i] += float(l.split(" ")[-2])
                            self.counts["inf_time"][i] += 1
                        if "Performed sorting of 10000 compounds" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['sort_time'][i] += float(l.split(':')[-1].split(',')[0].split('=')[-1])
                            self.counts['sort_time'][i] += 1
                            self.stats['sort_io_time'][i] += float(l.split(':')[-1].split(',')[1].split('=')[-1])
                            self.counts['sort_io_time'][i] += 1
                        if "Performed docking simulations in" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats["sim_time"][i] += float(l.split(" ")[-2])
                            self.counts["sim_time"][i] += 1
                        if "Performed docking simulation:" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['sim_io_time'][i] += float(l.split(':')[-1].split(',')[1].split('=')[-1])
                            self.counts['sim_io_time'][i] += 1
                        if "Performed training of" in l or "Performed training:" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['train_time'][i] += float(l.split(':')[-1].split(',')[0].split('=')[-1])
                            self.counts['train_time'][i] += 1
                            self.stats['train_io_time'][i] += float(l.split(':')[-1].split(',')[1].split('=')[-1])
                            self.counts['train_io_time'][i] += 1
                        
                if len(inf_io_times) > 0:
                    self.stats['inf_io_time'][i] = sum(inf_io_times)/len(inf_io_times)
                    self.counts['inf_io_time'][i] += 1
                else:
                    self.stats['inf_io_time'][i] = 0
                    self.counts['inf_io_time'][i] += 1


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

system = "aurora"
if system == "aurora":
    root = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25'
elif system == "local":
    root = '/Users/riccardobalin/Documents/ALCF/Conferences/PASC25'

nodes = [12, 48, 192, 768]
base_path = root+'/runs/pfs_vs_ddict'
cases = ['tiny_8192','small_32768','med_131072','full_500354']
wf_1ccs = WFStats(base_path,cases,nodes)
wf_1ccs.parse_files()

base_path = root+'/runs/full_wf'
cases = ['8192','32768','131072','500354']
wf_4ccs = WFStats(base_path,cases,nodes)
wf_4ccs.parse_files()
print(wf_4ccs.stats)

"""
# Plot component times
labels = ['Loading', 'Inference', 'Sorting', 'Simulation', 'Training']
x = np.arange(len(labels))  # the label locations
width = 0.12  # the width of the bars
factors = [-1.5,-0.5,0.5,1.5]
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
for i in range(len(wf.cases)):
    axs.bar(x+factors[i]*width, [wf.stats["load_time"][i], wf.stats["inf_time"][i], wf.stats["sort_time"][i], wf.stats["sim_time"][i], wf.stats["train_time"][i]], width,label=str(wf.nodes[i]))
#axs.set_yscale('log')
axs.set_ylabel('Time [sec]')
axs.set_title('Component Run Time')
axs.set_xticks(x);axs.set_xticklabels(labels)
#axs.set_xticks(x, labels)
axs.legend()
axs.grid(axis='y')
fig.savefig('plt_full_comp_time_bar.png')
"""

# Plot component times
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
#axs.plot(wf_1ccs.nodes, wf_1ccs.stats['load_time'],label = "Loading",marker="o",ls="-",markersize=10, linewidth=2)
#axs.plot(wf_1ccs.nodes, wf_1ccs.stats['inf_time'],label = "Inference",marker="o",ls="-",markersize=10, linewidth=2)
#axs.plot(wf_1ccs.nodes, wf_1ccs.stats['sort_time'],label = "Sorting",marker="o",ls="-",markersize=10, linewidth=2)
#axs.plot(wf_1ccs.nodes, wf_1ccs.stats['sim_time'],label = "Simulation",marker="o",ls="-",markersize=10, linewidth=2)
#axs.plot(wf_1ccs.nodes, wf_1ccs.stats['train_time'],label = "Training",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(wf_4ccs.nodes, wf_4ccs.stats['load_time'],label = "Loading",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(wf_4ccs.nodes, wf_4ccs.stats['inf_time'],label = "Inference",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(wf_4ccs.nodes, wf_4ccs.stats['sort_time'],label = "Sorting",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(wf_4ccs.nodes, wf_4ccs.stats['sim_time'],label = "Simulation",marker="o",ls="-",markersize=10, linewidth=2)
#axs.plot(wf_4ccs.nodes, wf_4ccs.stats['train_time'],label = "Training",marker="o",ls="-",markersize=10, linewidth=2)
files = ['8192', '32768', '131072','500354']
for i in range(len(wf_4ccs.nodes)):
    axs.text(wf_4ccs.nodes[i], wf_4ccs.stats['inf_time'][i]- 500, f"{files[i]} files", fontsize=13, color='black',fontweight='bold')

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(10,1000)
axs.set_ylabel('Time [sec]')
axs.set_xlabel('Number of Nodes')
axs.set_title('Workflow Component Run Time')
#axs.set_xticks(x, labels)
axs.legend(loc='lower right', ncol=2)
axs.grid()
axs.set_ylim(10,5000)
fig.savefig('plt_full_comp_time.png')

"""
# Plot efficiencies
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.plot(wf.nodes, wf.stats['load_time'][0]/wf.stats['load_time']*100,label = "Loading",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(wf.nodes, wf.stats['inf_time'][0]/wf.stats['inf_time']*100,label = "Inference",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(wf.nodes, wf.stats['sort_time'][0]/wf.stats['sort_time']*100,label = "Sorting",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(wf.nodes, wf.stats['sim_time'][0]/wf.stats['sim_time']*100,label = "Simulation",marker="o",ls="-",markersize=10, linewidth=2)
#axs.plot(wf.nodes, wf.stats['train_time']/wf.stats['train_time'][0]*100,label = "Training",marker="o",ls="-",markersize=10, linewidth=2)
axs.set_ylim(0,110)
axs.set_xscale('log')
axs.set_xlim(10,1000)
axs.set_ylabel('Time [sec]')
axs.set_ylabel('Number of Nodes')
axs.set_title('Component Run Time')
#axs.set_xticks(x, labels)
axs.legend()
axs.grid()
fig.savefig('plt_full_comp_eff.png')
"""


