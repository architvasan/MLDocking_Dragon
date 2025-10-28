from matplotlib import pyplot as plt
import glob

files = glob.glob("5*/test.out")+glob.glob("1*/test.out")
print(files)

# Loaded file 25 of size 1596 bytes in 1.6496051102876663e-07s, total size loaded: 4.494376480579376e-05

fig = plt.figure()
rate_ax = fig.add_subplot(2,1,1)
util_ax = fig.add_subplot(2,1,2)

for fp in files:
    lines = []
    t_elapsed = [0]
    pool_time = []
    max_pool_util = []
    min_pool_util = []
    num_files = 0
    with open(fp,'r') as f:
        lines = f.readlines()

    for ln in lines:
        if "Reading smiles for " in ln:
            target_num = int(ln.split()[-1])
        if "Loaded file " in ln:
            ln_split = ln.split()
            delta_t = float(ln_split[8][0:-2])/60.
            t_elapsed.append(t_elapsed[num_files]+delta_t)
            num_files += 1
        if "DDictManagerStats" in ln:
            stats_list = ln.split()
            pool_util = []
            for sl in stats_list:
                if "pool_utilization=" in sl:
                    pool_util.append(float(sl.split('=')[1][:-1]))
            max_pool_util.append(max(pool_util))
            min_pool_util.append(min(pool_util))
            pool_time.append(t_elapsed[num_files])


    rate = [1/(t_elapsed[i+1]-t_elapsed[i]) for i in range(len(t_elapsed)-1)]

    #plt.plot(range(len(t_elapsed[1:])),rate,label=fp)
    line,=rate_ax.plot(t_elapsed,
            [i for i in range(num_files+1)],
            label=fp)
    util_ax.plot(pool_time, max_pool_util, '--', c=line.get_color())
    util_ax.plot(pool_time, min_pool_util, ':', c=line.get_color())
    rate_ax.set_ylabel('Number of files loaded')
    util_ax.set_ylabel('min,max Pool utilization')
    rate_ax.set_xlabel('Time elapsed (minutes)')
    util_ax.set_xlabel('Time elapsed (minutes)')
    #plt.title('Time elapsed loading files')
    rate_ax.legend(fontsize='small')
fig.tight_layout()
fig.savefig('telapsed.png')
