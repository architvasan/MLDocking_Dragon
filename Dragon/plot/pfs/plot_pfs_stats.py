import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)


class PFSStats:
    def __init__(self,path):
        self.base_path = path
        self.n_nodes = 1
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
            'num_smiles': np.zeros((self.n_nodes,)),
            'iter_time': np.zeros((self.n_nodes,)),
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
            'num_smiles': np.zeros((self.n_nodes,)),
            'iter_time': np.zeros((self.n_nodes,)),
        }

    def avg(self, my_list):
        return sum(my_list)/len(my_list)

    def parse_files(self):
        for i in range(self.n_nodes):
            path = self.base_path+f"/mldocking*"
            # Loop over runs found
            run_files = glob.glob(path)
            for run_file in run_files:
                inf_io_times = []
                num_smiles = []
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
                            parsed_l_2 = l.split(":")[0]
                            num_smiles.append(int(parsed_l_2.split(" ")[6]))
                        if "Performed inference in" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats["inf_time"][i] = float(l.split(" ")[-2])
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
                            self.stats["sim_time"][i] = float(l.split(" ")[-2])
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
                        if "Performed iter" in l:
                            self.stats["iter_time"][i] = float(l.split(" ")[4])
                            self.counts['iter_time'][i] += 1
                        

                self.stats['inf_io_time'][i] = sum(inf_io_times)/len(inf_io_times)
                self.counts['inf_io_time'][i] += 1
                self.stats['num_smiles'][i] = sum(num_smiles)
                self.counts['num_smiles'][i] += 1


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

base_path = root+'/runs/pfs_vs_ddict/small_32768/pfs'
pfs = PFSStats(base_path)
pfs.parse_files()
print(pfs.stats['num_smiles'])
print()

base_path = root+'/runs/pfs_vs_ddict/small_32768/ddict'
ddict = PFSStats(base_path)
ddict.parse_files()
print(ddict.stats['num_smiles'])
print()


# Plot component times
labels = ['Inference', 'Sorting', 'Simulation', 'Training','Iteration']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
factors = [-0.5,0.5]
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.bar(x-0.5*width, [pfs.stats["inf_time"].item(), pfs.stats["sort_time"].item(), pfs.stats["sim_time"].item(), pfs.stats["train_time"].item(), pfs.stats["iter_time"].item()], width,label="PFS")
axs.bar(x+0.5*width, [ddict.stats["inf_time"].item(), ddict.stats["sort_time"].item(), ddict.stats["sim_time"].item(), ddict.stats["train_time"].item(), ddict.stats["iter_time"].item()], width,label="DDict")
axs.set_yscale('log')
axs.set_ylabel('Time [sec]')
axs.set_title('Component Run Time')
axs.set_xticks(x);axs.set_xticklabels(labels)
#axs.set_xticks(x, labels)
axs.legend()
axs.grid(axis='y')
axs.set_ylim(10,6000)
fig.savefig('plt_pfs_comp_time.png')

# Plot IO times
labels = ['Inference', 'Sorting', 'Simulation', 'Training']
x = np.arange(len(labels))  # the label locations
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.bar(x-0.5*width, [pfs.stats["inf_io_time"].item(), pfs.stats["sort_io_time"].item(), pfs.stats["sim_io_time"].item(), pfs.stats["train_io_time"].item()], width,label="PFS")
axs.bar(x+0.5*width, [ddict.stats["inf_io_time"].item(), ddict.stats["sort_io_time"].item(), ddict.stats["sim_io_time"].item(), ddict.stats["train_io_time"].item()], width,label="DDict")
axs.set_yscale('log')
axs.set_ylabel('Time [sec]')
axs.set_title('Component IO Time')
axs.set_xticks(x);axs.set_xticklabels(labels)
#axs.set_xticks(x, labels)
axs.legend()
axs.grid(axis='y')
axs.set_ylim(0.01,200)
fig.savefig('plt_pfs_io_time.png')

