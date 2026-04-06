"""Modularized helpers for imitate_episodes training.

Package layout:
- utils.py: argparse/options + shared helpers + common trajectory utils
- perturbation.py: online error injection and perturbation helpers
- correction.py: correction trajectory generation, EVAC rollout, export/debug
- training.py: policy/model setup, training loop, top-level main entry
"""
