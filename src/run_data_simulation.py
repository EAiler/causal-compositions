"""
Run file to make data simulations

----
Simulated Data is stored as a pickle file and can later be reused

"""

from typing import Dict, Text, Union

import json
import os

from absl import app
from absl import flags
from absl import logging

from datetime import datetime
import jax.numpy as np

# some more imports
import pickle
import time
import jax
import skbio.stats.composition as cmp

from simulate_data_fct import sim_IV_negbinomial

Results = Dict[Text, Union[float, np.ndarray]]

# Don't be afraid of *many* command line arguments.
# Everything should be a flag!
# ------------------------------- DUMMY PARAMS --------------------------------
flags.DEFINE_enum("enum", "choice1",
                  ("choice1", "choice2", "choice3"),
                  "An enum parameter.")
flags.DEFINE_string("string", "test", "String parameter.")
flags.DEFINE_bool("bool", False, "Boolean parameter.")
flags.DEFINE_float("float", 0.0, "Float parameter.")
flags.DEFINE_integer("integer", 0, "Integer parameter.")
# ---------------------------- INPUT/OUTPUT -----------------------------------
# I always keep those the same!
flags.DEFINE_string("data_dir", "../param_input/",
                    "Directory of the input data.")
flags.DEFINE_string("output_dir", "../results/",
                    "Path to the output directory (for results).")
flags.DEFINE_string("output_name", "",
                    "Name for result folder. Use timestamp if empty.")
flags.DEFINE_bool("plot", True,
                  "Whether to store plots while running.")
# ------------------------------ MISC -----------------------------------------
# I always keep this one!
flags.DEFINE_integer("seed", 0, "The random seed.")
FLAGS = flags.FLAGS

flags.DEFINE_string("paramfile", "paramFile_1", "name of python parameter file.")
flags.DEFINE_integer("n", 10000, "sample number.")
flags.DEFINE_integer("p", 10, "number of compositions/microbes.")
flags.DEFINE_integer("num_inst", 10, "number of instruments.")
flags.DEFINE_integer("num_star", 250, "number of interventional dataset.")

# local functions

# =============================================================================
# MAIN
# =============================================================================

def main(_):
    # ---------------------------------------------------------------------------
    # Directory setup, save flags, set random seed
    # ---------------------------------------------------------------------------
    FLAGS.alsologtostderr = True

    if FLAGS.output_name == "":
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        dir_name = FLAGS.output_name
    out_dir = os.path.join(os.path.abspath(FLAGS.output_dir), dir_name)


    logging.info(f"Save all output to {out_dir}...")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    FLAGS.log_dir = out_dir
    logging.get_absl_handler().use_absl_log_file(program_name="run")

    logging.info("Save FLAGS (arguments)...")
    with open(os.path.join(out_dir, 'flags.json'), 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

    logging.info(f"Set random seed {FLAGS.seed}...")
    # Depending on whether we are using JAX, just numpy, or something else
    # key = random.PRNGKey(FLAGS.seed)
    # np.random.seed(FLAGS.seed)

    # ---------------------------------------------------------------------------
    # Load parameter data
    # ---------------------------------------------------------------------------
    paramfile = FLAGS.paramfile
    inputfolder = FLAGS.data_dir

    n = FLAGS.n
    num_star = FLAGS.num_star
    p = FLAGS.p
    num_inst = FLAGS.num_inst

    param = __import__(paramfile)

    V = cmp._gram_schmidt_basis(p)
    alpha0 = np.hstack(
        [param.alpha0, jax.random.choice(param.key, np.array([1, 2, 2]), (p - 8,))])

    if num_inst == p:
        print("We can simulate a one on one relation between microbiomes and number of instruments")
        alphaT = np.diag(np.ones(p))
    else:
        print("Random simulation of relation between microbes and instruments")
        alphaT = jax.random.choice(param.key, np.array([0, 0, 0, 10]), (num_inst, p))

    mu_c = np.hstack([param.mu_c, jax.random.uniform(param.key, (p - 4,), minval=0.01, maxval=0.05)])
    mu_c = mu_c / mu_c.sum()  # has to be a compositional vector

    beta0 = param.beta0
    c_X = param.c_X

    betaT = np.hstack([param.betaT,
                       np.zeros((p - 8))])  # beta is chosen to sum up to one
    betaT_log = betaT
    c_Y = np.hstack([param.c_Y,
                     np.zeros((p - 12))])
    ps = np.hstack([np.zeros((int(p / 2),)), 0.8 * np.ones((p - int(p / 2),))])
    # ---------------------------------------------------------------------------
    # Simulate data
    # ---------------------------------------------------------------------------

    logging.info(f"Start data simulation.")
    start = time.time()
    confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_negbinomial(param.key,
                                                                         n=n,
                                                                         p=p,
                                                                         num_inst=num_inst,
                                                                         mu_c=mu_c,
                                                                         c_X=c_X,
                                                                         alpha0=alpha0,
                                                                         alphaT=alphaT,
                                                                         c_Y=c_Y,
                                                                         beta0=beta0,
                                                                         betaT=betaT,
                                                                         num_star=num_star,
                                                                         ps=ps
                                                                         )
    time_elapsed = np.round((time.time() - start)/60, 2)
    logging.info(f"This took {time_elapsed} minutes to finish.")
    # ---------------------------------------------------------------------------
    # Save result dictionary
    # ---------------------------------------------------------------------------
    paramdict = {
        "n": n,
        "p": p,
        "num_inst": num_inst,
        "c_X": c_X,
        "alpha0": alpha0,
        "alphaT": alphaT,
        "beta0": beta0,
        "betaT": betaT,
        "mu_c": mu_c,
        "c_Y": c_Y,
        "betaT_log": betaT_log,
        "ps": ps,
        "V": V,
        "key": param.key
    }
    res = {
        "paramfile": paramfile,
        "paramdict": paramdict,
        "confounder": confounder,
        "Z_sim": Z_sim,
        "X_sim": X_sim,
        "Y_sim": Y_sim,
        "X_star": X_star,
        "Y_star": Y_star
    }


    with open(os.path.join(inputfolder, "data_simulation_n" + str(n) + "_p" + str(p) + "_k" + str(num_inst) + ".pickle"), "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


    logging.info(f"DONE")


if __name__ == "__main__":
    app.run(main)
