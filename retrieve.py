import subprocess
import pathlib
import argparse

remote_path = pathlib.Path("/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Ian/code/cplAE_MET/data")
local_path = pathlib.Path("/Users/ian.convy/code/cplAE_MET/data")

def get_experiment(exp_dir, dest_dir):
    target_path = remote_path / exp_dir
    dest_path = local_path / dest_dir
    uri = f"ian.convy@hpc-login:{target_path}"
    subprocess.run(["scp", "-r", uri, dest_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("dest_dir")
    args = parser.parse_args()
    get_experiment(args.exp_dir, args.dest_dir)