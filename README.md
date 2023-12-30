# AMPG

The official implementation of the paper "Adaptive Message Passing Mechanism for Graph Neural Networks".

## The overall Architecture of AMPG
<p align="middle">
<img src="pics/Architecture.png" width="600">
</p>

## Dependencies

This is the list of the package versions required for our experiments.

```txt
python==3.8.18
pytorch==1.13.1
torch_geometric==2.4.0
torch_sparse==0.6.15
torch_scatter==2.1.0
torch_cluster==1.6.0
torch_spline_conv==1.2.1
wandb==0.15.12
```


## Run

We manage our experiments with [wandb](https://wandb.ai), to reproduce the results we reported in our paper, please follow these steps:

- Set up the environment variables. Below 2 environment variables `$YOUR_WANDB_ENTITY$` and `$YOUR_WANDB_PROJECT$` are your wandb username and the name of the project.
    ```bash
    export WANDB_entity=$YOUR_WANDB_ENTITY$
    export WANDB_project=$YOUR_WANDB_PROJECT$
    ```

- Choose best hyper-parameters you want to run with, and create wandb sweep with that file.
    
    We record the best hyper-parameters in folder `best_params`, you can index corresponding file by name.
    ```bash
    python sweep.py --sweep_file=best_params/Squirrel_AMPG.yaml
    ```

- You will get an sweep ID `$SWEEP_ID$` and sweep URL `$SWEEP_URL$` from last step, like:
    ```bash
    Create sweep with ID: $SWEEP_ID$
    Sweep URL: $SWEEP_URL$
    ```

    then run below command will start runs with GPU. Parameter `$INDEX_GPU$:$PARALLEL_RUNS$` indicate we will run `$PARALLEL_RUNS$` runs in parallel with GPU `$INDEX_GPU$`.

    ```bash
    python agents.py --sweep_id=$SWEEP_ID$ --gpu_allocate=$INDEX_GPU$:$PARALLEL_RUNS$
    ```

- You can check the results in `$SWEEP_URL$`, a website hosted on [wandb.ai](https://wandb.ai).


