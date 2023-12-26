"""Run for multiple hyperparameter values on slurm cluster."""

import json
import os
from copy import deepcopy
from itertools import product
from typing import Text, Sequence, Dict, Any

from absl import app
from absl import flags

flags.DEFINE_string("experiment_name", None,
                    "The name of the experiment (used for output folder).")
flags.DEFINE_string("result_dir", "/home/haicu/elisabath.ailer/Projects/CausalComp/Output/",
                    "Base directory for all results.")
flags.DEFINE_bool("gpu", True, "Whether to use GPUs.")

# FIXED FLAGS
flags.DEFINE_float("logcontrast_threshold", 0.7, "Log Contrast Threshold Value.")
flags.DEFINE_float("kernel_alpha", 1, "Penalization parameter for KIV second stage.")
flags.DEFINE_integer("num_runs", 50, "Number of runs for confidence interval computation.")
flags.DEFINE_list("selected_methods", ["NoComp", "OnlyLogContrast", "DIR+LC", "ILR+LC", "ALR+LC", "ILR+ILR",
                                       "KIV", "M_KIV", "OnlyKIV"],
                  "Define the methods that should be run.")
flags.DEFINE_bool("use_data_in_folder", False, "Check if data is already available in the folder, if yes, then use the pickle files in the folder.")
flags.DEFINE_string("add_id", "", "additional identifier, if data should be reused for another method etc.")
# DEFINITION OF PARAMETER GRID
flags.DEFINE_list("paramfile", ["linear_n100_p250_q10",
                                "linear_n1000_p3_q2",
                                "negbinom_n10000_p250_q10",
                                "negbinom_n10000_p30_q10",
                                "negbinom_n1000_p3_q2",
                                "nonlinear_n1000_p3_q2",
                                "linear_n10000_p250_q10",
                                "linear_n10000_p30_q10",
                                "linear_weak_n1000_p3_q2"],
                  "name of python file with the parameters for simulation.")
flags.mark_flag_as_required("experiment_name")
FLAGS = flags.FLAGS


# Some values and paths to be set
user = "elisabath.ailer"  # TODO : check that this is not elisabath.ailer (!!!) -> there were some mistakes made
project = "CausalComp"
executable = f"/home/haicu/{user}/miniconda3/envs/comp_iv/bin/python"  # inserted the respective environment
run_file = f"/home/haicu/{user}/Projects/{project}/src/run_methods_confidence.py"  # MAY NEED TO UPDATE THIS


# Specify the resource requirements *per run*
num_cpus = 4
num_gpus = 1
mem_mb = 16000
max_runtime = "00-24:00:00"


def get_output_name(value_dict: Dict[Text, Any]) -> Text:
  """Get the name of the output directory."""
  name = ""
  for k, v in value_dict.items():
    name += f"-{k}_{v}"
  return name[1:]


def get_flag(key: Text, value: Any) -> Text:
  if isinstance(value, bool):
    return f' --{key}' if value else f' --no{key}'
  if isinstance(value, list):
    command = f' --{key}='
    for i in range(len(value)-1):
      command += f'"{value[i]}",'
    command += f'"{value[-1]}"'
    return command
  else:
    return f' --{key}={value}'


def submit_all_jobs(args: Sequence[Dict[Text, Any]], fixed_flags) -> None:
  """Genereate submit scripts and launch them."""
  # Base of the submit file
  base = list()
  base.append(f"#!/bin/bash")
  base.append("")
  base.append(f"#SBATCH -J {project}{'_gpu' if FLAGS.gpu else ''}")
  base.append(f"#SBATCH -c {num_cpus}")
  base.append(f"#SBATCH --mem={mem_mb}")
  base.append(f"#SBATCH -t {max_runtime}")
  base.append(f"#SBATCH --nice=10000")
  if FLAGS.gpu:
    base.append(f"#SBATCH -p gpu_p")
    base.append(f"#SBATCH --qos gpu_long") #gpu_long
    base.append(f"#SBATCH --gres=gpu:{num_gpus}")
    #base.append(f"#SBATCH --exclude=icb-gpusrv0[1]")  # keep for interactive
    #base.append(f"#SBATCH --exclude=icb-gpusrv0[22-25]")  # keep for interactive
  else:
    base.append(f"#SBATCH -p cpu_p")

  for i, arg in enumerate(args):
    lines = deepcopy(base)
    output_name = get_output_name(arg)

    # Directory for slurm logs
    result_dir = os.path.join(FLAGS.result_dir, FLAGS.experiment_name)
    logs_dir = os.path.join(result_dir, output_name)

    # Create directories if non-existent (may be created by the program itself)
    if not os.path.exists(logs_dir):
      os.makedirs(logs_dir)

    # The output, logs, and errors from running the scripts
    logs_name = os.path.join(logs_dir, "slurm")
    lines.append(f"#SBATCH -o {logs_name}.out")
    lines.append(f"#SBATCH -e {logs_name}.err")

    # Queue job
    lines.append("")
    runcmd = executable
    runcmd += " "
    runcmd += run_file
    # ASSUMING RUNFILE TAKES THESE THREE ARGUMENTS
    runcmd += f' --output_dir {result_dir}'
    runcmd += f' --output_name {output_name}'

    # Sweep arguments
    for k, v in arg.items():
      runcmd += get_flag(k, v)

    # Fixed arguments
    for k, v in fixed_flags.items():
      runcmd += get_flag(k, v)

    lines.append(runcmd)
    lines.append("")
    print(lines)
    # Now dump the string into the `run_all.sub` file.
    with open("run_job.cmd", "w") as file:
      file.write("\n".join(lines))

    print(f"Submitting {i}...")
    os.system("sbatch run_job.cmd")


def main(_):
  """Initiate multiple runs."""

  sweep = {
      "paramfile": FLAGS.paramfile
  }

  values = list(sweep.values())
  args = list(product(*values))
  keys = list(sweep.keys())
  args = [{keys[i]: arg[i] for i in range(len(keys))} for arg in args]
  n_jobs = len(args)
  sweep_dir = os.path.join(FLAGS.result_dir, FLAGS.experiment_name)

  fixed_flags = {
    "selected_methods": FLAGS.selected_methods,
    "logcontrast_threshold": FLAGS.logcontrast_threshold,
    "kernel_alpha": FLAGS.kernel_alpha,
    "num_runs": FLAGS.num_runs,
    "use_data_in_folder": FLAGS.use_data_in_folder,
    "add_id": FLAGS.add_id
  }

  # Create directories if non-existent
  if not os.path.exists(sweep_dir):
    os.makedirs(sweep_dir)
  print(f"Store sweep dictionary to {sweep_dir}...")
  with open(os.path.join(sweep_dir, "sweep.json"), 'w') as fp:
    json.dump(sweep, fp, indent=2)

  print(f"Generate all {n_jobs} submit script and launch them...")
  submit_all_jobs(args, fixed_flags)

  print(f"DONE")


if __name__ == "__main__":
  app.run(main)
