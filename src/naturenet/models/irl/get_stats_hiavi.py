"""
HIAVI K-fold CV (+ optional final holdout run) for every (K, run): K in [min_latent, max_latent] inclusive,
seed = seed + run, data / aligned arrays / per-fold T computed once and shared. --n_hidden / --run / --fold
select one piece per job-array task; finished folds are skipped; ll_hiavi.csv collects train / test LL.
Set run_hiavi_final: true to also run the holdout fit for each (K, run).
"""
import argparse

from naturenet.models.irl.run_hiavi_kfold import prepare_hiavi_data, run_hiavi_final, run_hiavi_kfold
from naturenet.models.irl.swirl_data import iter_loop_configs, load_swirl_dataset

from sit_fuse.utils import read_yaml


def parse_args():  # same flags as get_stats_nn.py (not imported: that module pulls in JAX)
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--n_hidden", type=int, default=None, help="Only this number of hidden modes.")
    parser.add_argument("--run", type=int, default=None, help="Only this run index.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only these fold indices.")
    return parser.parse_args()


def loop_hiavi(yml_conf, args):
    data = prepare_hiavi_data(yml_conf, load_swirl_dataset(yml_conf, allow_variable=True))
    for conf in iter_loop_configs(yml_conf, n_hidden=args.n_hidden, run=args.run):
        print(conf["run_uid"], "K", conf["n_hidden"], "seed", conf["seed"])
        if conf.get("run_hiavi_kfold", True):
            run_hiavi_kfold(conf, data=data, folds_to_run=args.fold)
        if conf.get("run_hiavi_final", False) and args.fold is None:
            run_hiavi_final(conf, data=data)


if __name__ == '__main__':
    args = parse_args()
    loop_hiavi(read_yaml(args.yaml), args)
