
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
NN-SWIRL S-1 K-fold for every (K, run). Run get_stats_nn.py first (per-fold NN ARHMM init).
Same loop as get_stats.py: K inclusive, seed + run, out_dir from YAML, data / one-hots / per-fold T built
once and reused across (K, run), skip_n_states (list, default none) replaces the hard-coded n_states == 11
skip, --n_hidden / --run / --fold for job arrays.
"""
from naturenet.models.irl.run_swirl_nn_kfold import run_swirl_nn
from naturenet.models.irl.run_swirl_nn_kfold import prepare_nn_data
from naturenet.models.irl.get_stats_nn import parse_args
from naturenet.models.irl.swirl_data import as_list, iter_loop_configs, load_swirl_dataset

from sit_fuse.utils import read_yaml


def loop_swirl(yml_conf, args):
    if int(yml_conf["n_states"]) in [int(s) for s in as_list(yml_conf.get("skip_n_states"))]:
        print("n_states", yml_conf["n_states"], "is in skip_n_states; nothing to do")
        return
    data = prepare_nn_data(yml_conf, load_swirl_dataset(yml_conf))
    for conf in iter_loop_configs(yml_conf, n_hidden=args.n_hidden, run=args.run):
        print(conf["run_uid"], "seed", conf["seed"])
        run_swirl_nn(conf, data=data, folds_to_run=args.fold)


if __name__ == '__main__':
    args = parse_args()
    loop_swirl(read_yaml(args.yaml), args)
