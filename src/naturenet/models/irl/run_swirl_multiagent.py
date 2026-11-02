"""
Multi-agent mini-test with the SWIRL backend (same design, outputs and plots as run_hiavi_multiagent).

    python -m naturenet.models.irl.run_swirl_multiagent -y config/swirl/swirl_multiagent.yaml [--K 2] [--skip_null]
    python -m naturenet.models.irl.plot_hiavi_multiagent -y config/swirl/swirl_multiagent.yaml
    python -m naturenet.models.irl.plot_multiagent_compare -y config/swirl/swirl_multiagent.yaml \
        --subdirs multiagent multiagent_swirl

Outputs go to <out_dir>/<multiagent.ma_subdir or multiagent_swirl>/.
"""
from naturenet.models.irl.run_hiavi_multiagent import cli

if __name__ == "__main__":
    cli("swirl")
