# Install

## Aurora
Run this script in the directory where you want the environment installed:
```shell
./install_aurora.sh
```

Edit `env_aurora.sh` to include the path to the environment and run to activate:
```shell
./env_aurora.sh
```

# Sequential Workflow

Get in an interactive session with several nodes or from a submit script:

```shell
./run_driver_seq.sh 256
```

Edit the integer to match the number of files for the workflow.  You may also need to edit paths in the script.

To submit a job
```
qsub sub_driver_seq_aurora.sh
```
