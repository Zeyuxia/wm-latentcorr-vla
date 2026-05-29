#!/bin/bash
set -euo pipefail
cd /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_multitask_eval_ep100_gpu4567_finalrollout_30.sh
/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_multitask_eval_ep150_gpu4567_finalrollout_30.sh
