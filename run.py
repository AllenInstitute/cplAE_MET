import argparse
import yaml
import subprocess
import os
import shutil
import pathlib
import sys

def safe_unlink(path):
    if path.exists():
        path.unlink()

def create_sbatch_script(python_file, exp_dir, config_path, slurm):
    exp_name = exp_dir.name
    output_path = exp_dir / "terminal.out"
    with open(exp_dir / "script.sh", "w") as target:
        script_string = "\n".join([
            "#!/bin/bash", f"#SBATCH --partition={slurm['partition']}",
            f"#SBATCH --job-name={exp_name}", f"#SBATCH --cpus-per-task={slurm['cpus']}",
            f"#SBATCH --gpus={slurm['gpus']}", f"#SBATCH --nodes={slurm['nodes']}",
            f"#SBATCH --mem={slurm['memory']}", f"#SBATCH --time={slurm['time']}",
            f"#SBATCH --output={output_path}", "", f"cd {slurm['directory']}",
            f"source activate {slurm['conda']}", "", f"python -u {python_file} {exp_name} {config_path}" 
        ])
        target.write(script_string)

def get_git_hash():
    try:
        git_hash = subprocess.run(
            (["powershell"] if os.name == "nt" else []) + ["git", "rev-parse", "--short", "HEAD"],
            capture_output = True 
        ).stdout.decode().strip()
    except Exception:
        git_hash = ""
    return git_hash

def record_settings(exp_dir, config_path):
    git_hash = get_git_hash()
    if git_hash:
        with open(exp_dir / "git_hash.txt", "w") as target:
            target.write(git_hash)
    else:
        print("Git hash not saved.")
    shutil.copy(config_path, exp_dir / "config.yaml")

def clear_experiment(exp_dir):
    safe_unlink(exp_dir / "config.yaml")
    safe_unlink(exp_dir / "git_hash.txt")
    safe_unlink(exp_dir / "terminal.out")
    safe_unlink(exp_dir / "script.sh")
    safe_unlink(exp_dir / "model.pt")
    safe_unlink(exp_dir / "best_params.pt")
    safe_unlink(exp_dir / "outputs.npz")
    tensorboard_path = exp_dir / "tn_board"
    if tensorboard_path.exists():
        for path in tensorboard_path.iterdir():
            if path.is_file():
                path.unlink()
            else:
                for path2 in path.iterdir():
                    path2.unlink()

if __name__ == "__main__":
    # Experiment name and path to the config YAML file must be provided.
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help = "Name of experiment.")
    parser.add_argument("config_file", help = "Path to configuration YAML file.")
    parser.add_argument("--local", action = "store_true")
    args = parser.parse_args()

    # Config YAML file contains the specific parameters for the experiment.
    with open(args.config_file, "r") as target:
        config = yaml.safe_load(target)

    # The specific Python file to be run depends on the experiment type.
    if config["experiment"] == "morphology":
        python_file = "train_morphology.py"
    elif config["experiment"] == "align":
        python_file = "alignment.py"
    elif set(config["experiment"]) - set("met"):
        raise ValueError(f'''Experiment "{config['experiment']}" not recognized.''')
    else:
        python_file = "train.py"
    
    # Define directory for experiment outputs and check if it already exists.
    exp_dir = pathlib.Path(config["output_dir"]) / args.exp_name
    if exp_dir.exists():
        choice = input(f'Experiment "{args.exp_name}" already exists. Replace? (yes/no) ')
        if choice == "yes":
            clear_experiment(exp_dir)
        else:
            print("Exiting...")
            sys.exit()
    else:
        exp_dir.mkdir(parents = True)

    # The parameters are recorded in the experiment directory, and the Python file is run on SLURM (if not local).
    print(f'\nRunning experiment "{args.exp_name}":\n')
    record_settings(exp_dir, args.config_file)
    if not args.local:
        create_sbatch_script(python_file, exp_dir, args.config_file, config)
        subprocess.run(["sbatch", exp_dir / "script.sh"])
    else:
        subprocess.run(["python", "-u", python_file, args.exp_name, args.config_file])
