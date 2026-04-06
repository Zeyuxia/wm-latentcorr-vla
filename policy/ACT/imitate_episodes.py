import os
import sys

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"
# Required for deterministic CuBLAS operations
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evac'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from imitate_episodes_pkg.utils import build_parser
from imitate_episodes_pkg.training import main


if __name__ == "__main__":
    parser = build_parser()
    main(vars(parser.parse_args()))