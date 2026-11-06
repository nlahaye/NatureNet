
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
ARHMM K-fold init for every (K, run).

K runs over [min_latent, max_latent] INCLUSIVE; run i uses seed = seed + i (previously every run reused
the same seed). out_dir comes from the YAML (was hard-coded). Data are loaded once for all (K, run).
For a cluster job array, give --n_hidden / --run / --fold to run one piece per task; finished folds
are skipped, so re-submitting resumes.

Pseudo-count sweep: --pseudo_count 0.1 [--pseudo_support T] sets arhmm_pseudo_count / arhmm_pseudo_support and
appends _pc0.1 (or _pc0.1T) to run_uid, so each variant writes its own files and never touches the SWIRL init
files of the base run. Each fold also gets the action LL (train_ll_act / test_ll_act; arhmm_action_ll.py).
"""
import argparse

from naturenet.models.irl.run_arhmm_kfold import run_arhmm
from naturenet.models.irl.swirl_data import iter_loop_configs, load_swirl_dataset

from sit_fuse.utils import read_yaml


def pseudo_variant(yml_conf, pseudo_count=None, pseudo_support=None):
    """Copy of yml_conf for one pseudo-count variant (run_uid suffixed); unchanged if both are None."""
    if pseudo_count is None and pseudo_support is None:
        return yml_conf
    conf = dict(yml_conf)
    if pseudo_count is not None:
        conf["arhmm_pseudo_count"] = float(pseudo_count)
    if pseudo_support is not None:
        conf["arhmm_pseudo_support"] = None if pseudo_support in ("none", "all") else pseudo_support
    tag = f"{float(conf.get('arhmm_pseudo_count', 1e-5)):g}" + ("T" if conf.get("arhmm_pseudo_support") == "T" else "")
    conf["run_uid"] = f"{yml_conf['run_uid']}_pc{tag}"
    return conf


def loop_arhmm(yml_conf, args):
    yml_conf = pseudo_variant(yml_conf, getattr(args, "pseudo_count", None), getattr(args, "pseudo_support", None))
    data = load_swirl_dataset(yml_conf, allow_variable=True)
    for conf in iter_loop_configs(yml_conf, n_hidden=args.n_hidden, run=args.run):
        print(conf["run_uid"], "seed", conf["seed"])
        run_arhmm(conf, data=data, folds_to_run=args.fold)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--n_hidden", type=int, default=None, help="Only this number of hidden modes.")
    parser.add_argument("--run", type=int, default=None, help="Only this run index.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only these fold indices.")
    parser.add_argument("--pseudo_count", type=float, default=None,
                        help="ARHMM transition pseudo-count (default: YAML arhmm_pseudo_count, else 1e-5); suffixes run_uid.")
    parser.add_argument("--pseudo_support", choices=["T", "all"], default=None,
                        help="T: add the pseudo-count only to x -> x' moves reachable under the fold's T.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    loop_arhmm(yml_conf, args)
