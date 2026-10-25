
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
SWIRL K-fold (init + final) for every (K, run).

K runs over [min_latent, max_latent] INCLUSIVE; run i uses seed = seed + i (previously every run reused
the same seed). out_dir comes from the YAML (was hard-coded). Data are loaded once for all (K, run).
For a cluster job array, give --n_hidden / --run / --fold to run one piece per task; finished folds
are skipped, so re-submitting resumes.
"""
import argparse

from naturenet.models.irl.run_swirl_kfold import prepare_data, run_swirl_final, run_swirl_init
from naturenet.models.irl.swirl_data import iter_loop_configs, load_swirl_dataset

from sit_fuse.utils import read_yaml


def loop_swirl(yml_conf, args):
    data = load_swirl_dataset(yml_conf)
    data = prepare_data(yml_conf, data)
    for conf in iter_loop_configs(yml_conf, n_hidden=args.n_hidden, run=args.run):
        print(conf["run_uid"], "seed", conf["seed"])
        run_swirl_init(conf, data=data, folds_to_run=args.fold)
        run_swirl_final(conf, data=data, folds_to_run=args.fold)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--n_hidden", type=int, default=None, help="Only this number of hidden modes.")
    parser.add_argument("--run", type=int, default=None, help="Only this run index.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only these fold indices.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    loop_swirl(yml_conf, args)
