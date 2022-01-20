import sys
import os
import subprocess

import hydra
import omegaconf


def create_slurm_file(
    new_script_path,
    train_cmd,
    job_name,
    flags="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    mem=0,
    overcommit=True,
    nodes=1,
    ntasks_per_node=8,
    gpus_per_task=1,
    partition="batch",
    account=None,
):
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        if gpus_per_task is not None:
            f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if dependency is not None:
            if dependency != "singleton":
                dependency = f"afterany:{dependency}"
            f.writelines(f"#SBATCH --dependency={dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if account is not None:
            f.writelines(f"#SBATCH -A {account}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} sh -c "{train_cmd}"\n\n')
        f.writelines("set +x\n")


def create_bcp_submit_cmd(
    job_name,
    container,
    workspace_common,
    workspace_scripts,
    bignlp_path,
    bcp_script,
    instance,
    num_nodes,
    ntasks_per_node=8,
    array_type="pytorch",
    total_runtime="10h"
):
    base_cmd = f"cd {bignlp_path}; NGC_NTASKS_PER_NODE={ntasks_per_node} {bcp_script}"
    if (num_nodes == 1):
                num_nodes = 2  # bcprun needs at least 2 nodes    
    submit_cmd = f"ngc batch run --name \"{job_name}\" --image \"{container}\" \
    --commandline \"{base_cmd}\" --workspace {workspace_common}:/workspace-common \
    --workspace {workspace_scripts}:/workspace-scripts --result /result \
    --preempt RUNONCE --instance {instance} --replicas {num_nodes} \
    --array-type {array_type} --total-runtime {total_runtime}"
    
    return submit_cmd

def create_bcp_file(
    bignlp_path,
    train_cmd,
    num_nodes,
    log_file,
    err_file,
    new_script_path
):
    with open(new_script_path, "w") as f:
        # Replace bcprun by {bignlp_path}/bcprun2 if latest bcprun with local-rank fix is not deployed
        f.writelines(f'bcprun -n {num_nodes} -c \"{train_cmd}\" >> {log_file} 2>>{err_file} \n')
        f.writelines("\n")
        f.writelines("set +x \n") 
    os.chmod(new_script_path, 0o755)

def run_training(cfg, hydra_args="", dependency=None):
    # Read config
    bignlp_path = cfg.get("bignlp_path")
    container = cfg.get("container")
    train_cfg = cfg.get("training")
    run_cfg = train_cfg.get("run")
    megatron_cfg = train_cfg.get("megatron")

    # Run parameters
    name = run_cfg.get("name")
    results_dir = run_cfg.get("results_dir")
    log_dir = run_cfg.get("log_dir")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    new_script_path = os.path.join(bignlp_path, f"train_scripts/{name}.sh")
    code_path = os.path.join(bignlp_path, "train_scripts/pretrain_gpt.py")
    train_cmd = f"python3 -u {code_path} {hydra_args}"

    bcp_cfg = train_cfg.get("bcp")
    create_bcp_file(
        bignlp_path=bignlp_path,
        new_script_path=new_script_path,
        train_cmd=train_cmd,
        num_nodes=bcp_cfg.get("nodes"),
        log_file=f"{log_dir}/log.txt",
        err_file=f"{log_dir}/err.txt"
    )
    
    # BCP submit command
    num_nodes = bcp_cfg.get("nodes")
    ntasks_per_node = bcp_cfg.get("ntasks_per_node")
    gpus_per_task = bcp_cfg.get("gpus_per_task")
    instance = bcp_cfg.get("instance")
    time_limit = bcp_cfg.get("time_limit")

    submit_cmd = create_bcp_submit_cmd(
        job_name=bcp_cfg.get("job_name"),
        container=container,
        workspace_common=bcp_cfg.get("workspace_common"),
        workspace_scripts=bcp_cfg.get("workspace_scripts"),
        bignlp_path=bignlp_path,
        bcp_script=new_script_path,
        instance=instance,
        num_nodes=num_nodes,
        ntasks_per_node=ntasks_per_node,
        array_type="PYTORCH",
        total_runtime=time_limit
    )

    print(f"\n Submit command after data is ready:\n {submit_cmd}")
    print(f"\n Script file: {new_script_path}")
    
