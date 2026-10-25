
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
NN ARHMM K-fold init for every (K, run): K in [min_latent, max_latent] INCLUSIVE (was exclusive),
seed = seed + run (was the same seed for every run), out_dir from the YAML (was hard-coded), data loaded
once. --n_hidden / --run / --fold run one piece per job-array task; finished folds are skipped.
"""
import argparse

from naturenet.models.irl.run_arhmm_kfold_nn import run_arhmm
from naturenet.models.irl.swirl_data import iter_loop_configs, load_swirl_dataset

from sit_fuse.utils import read_yaml


def loop_arhmm(yml_conf, args):
    data = load_swirl_dataset(yml_conf)
    for conf in iter_loop_configs(yml_conf, n_hidden=args.n_hidden, run=args.run):
        print(conf["run_uid"], "seed", conf["seed"])
        run_arhmm(conf, data=data, folds_to_run=args.fold)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--n_hidden", type=int, default=None, help="Only this number of hidden modes.")
    parser.add_argument("--run", type=int, default=None, help="Only this run index.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only these fold indices.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    loop_arhmm(read_yaml(args.yaml), args)
