"""
HIAVI entry point (kept for existing scripts / job files). The implementation moved to hiavi_model.py
(model) and run_hiavi_kfold.py (K-fold CV + final run); this file runs, per the YAML:
  run_hiavi_kfold (default True): animal-grouped K-fold CV for n_hidden (or n_hidden_init);
  run_hiavi_final (default True): final holdout run (tag ""), the input for plot_hiavi.py.
Use get_stats_hiavi.py for the (K, run) sweep.

Bugs fixed vs. the original (which called hiql.algorithms.HIAVI):
- hiql's HIAVI.fit hard-codes 2 latents (randn(2), identity(2)): num_latents=8 failed to broadcast;
- hiql's backward pass aliases the beta array (in-place +=), double-counting emissions in gamma / xi;
- the best restart was picked by TEST LL (optimistic CV) -> picked by TRAIN LL;
- the reported "ll" was mean_t logsumexp(log gamma_t + log pi(a_t | s_t)) (posterior of the scored step,
  not a likelihood) -> forward log p(a | s) per step;
- `for num_trajs in arange(0, 8)` never subset the data (the same CV ran 8 times) and params were saved
  only for num_trajs == 4 (to fold dirs shared by every run);
- KFold(n_splits=len(trajs)) was leave-one-sequence-out mixing animals across train / test, with T built
  from all animals -> shared animal-grouped folds and per-fold T from the train animals;
- while-True EM with stochastic responsibilities could cycle forever -> tolerance / max_iters;
- gen_traj silently dropped sequences with <= 5 steps and kept invalid (-1) states; prints in loops,
  hard-coded GPU, global n_states, fixed np.random.seed(42) for every run.
"""
import argparse

from naturenet.models.irl.run_hiavi_kfold import prepare_hiavi_data, run_hiavi_final, run_hiavi_kfold
from naturenet.models.irl.swirl_data import load_swirl_dataset


def run_hiavi(yml_conf, folds_to_run=None):
    conf = dict(yml_conf)
    conf.setdefault("n_hidden", conf.get("n_hidden_init"))
    data = prepare_hiavi_data(conf, load_swirl_dataset(conf))
    if conf.get("run_hiavi_kfold", True):
        run_hiavi_kfold(conf, data=data, folds_to_run=folds_to_run)
    if conf.get("run_hiavi_final", True) and folds_to_run is None:
        run_hiavi_final(conf, data=data)


if __name__ == '__main__':
    from sit_fuse.utils import read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None)
    args = parser.parse_args()
    run_hiavi(read_yaml(args.yaml), args.fold)
