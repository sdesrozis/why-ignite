import os
import subprocess
import torch


if __name__ == '__main__':

    # general informations about pytorch
    print(f"torch version : {torch.version.__version__}")
    print(f"torch git version : {torch.version.git_version}")

    # general information about cuda
    if torch.cuda.is_available():
        print(f"torch version cuda : {torch.version.cuda}")
        print(f"number of cuda device(s) : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"- device {i} : {torch.cuda.get_device_properties(i)}")
    else:
        print("no cuda available")

    # general information about slurm
    if "SLURM_JOBID" in os.environ:
        for k in ["SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS", "SLURM_JOB_NODELIST"]:
            print(f"{k} : {os.environ[k]}")
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        print(f"hostnames : {hostnames}")
