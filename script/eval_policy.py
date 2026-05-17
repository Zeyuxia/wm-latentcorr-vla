import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
import re
import random

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
USE_COLOR = sys.stdout.isatty()


def colorize(text, code):
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_seed_list(seed_file):
    if seed_file is None:
        return None
    if not os.path.isfile(seed_file):
        raise FileNotFoundError(f"seed file not found: {seed_file}")

    with open(seed_file, "r", encoding="utf-8") as f:
        content = f.read()

    tokens = re.findall(r"-?\d+", content)
    if len(tokens) == 0:
        raise ValueError(f"no valid integer seed found in: {seed_file}")

    return [int(tok) for tok in tokens]


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_tag = str(usr_args.get("eval_tag", "")).strip()
    if eval_tag:
        safe_tag = re.sub(r"[^0-9A-Za-z._-]+", "_", eval_tag)
        current_time = f"{current_time}_{safe_tag}"
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    # Let CLI overrides win over task yaml for eval-time switches such as video logging.
    # argparse gives CLI values as strings, and bool("false") is True, so parse explicitly.
    if "eval_video_log" in usr_args and usr_args["eval_video_log"] is not None:
        value = usr_args["eval_video_log"]
        if isinstance(value, str):
            args["eval_video_log"] = value.strip().lower() in {"1", "true", "yes", "y"}
        else:
            args["eval_video_log"] = bool(value)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print(colorize("Messy Table:", "95") + " " + str(args["domain_randomization"]["cluttered_table"]))
    print(colorize("Random Background:", "95") + " " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print(colorize("Random Light:", "95") + " " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print(colorize("Random Table Height:", "95") + " " + str(args["domain_randomization"]["random_table_height"]))
    print(colorize("Random Head Camera Distance:", "95") + " " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print(colorize("Head Camera Config:", "94") + " " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print(colorize("Wrist Camera Config:", "94") + " " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print(colorize("Embodiment Config:", "94") + " " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]
    seed_file = usr_args.get("seed_file", None)
    seed_list = load_seed_list(seed_file)

    start_seed_override = usr_args.get("start_seed", None)
    if start_seed_override is not None:
        st_seed = int(start_seed_override)
    else:
        st_seed = 100000 * (1 + seed)
    initial_success_count = int(usr_args.get("initial_success_count", 0) or 0)
    initial_test_count = int(usr_args.get("initial_test_count", 0) or 0)
    suc_nums = []
    total_target_eval_num = len(seed_list) if seed_list is not None else 100
    test_num = total_target_eval_num
    if seed_list is None:
        test_num = max(0, int(total_target_eval_num) - int(initial_test_count))
    topk = 1

    if seed_list is not None:
        print(f"{colorize('Using seed file:', '96')} {seed_file}")
        print(f"{colorize('Total seeds for eval:', '96')} {test_num}")
    elif initial_test_count > 0:
        print(
            f"{colorize('Resume eval:', '96')} already_done={initial_test_count}, "
            f"remaining={test_num}, target_total={total_target_eval_num}"
        )

    model = get_model(usr_args)
    st_seed, suc_num, final_test_num = eval_policy(task_name,
                                                   TASK_ENV,
                                                   args,
                                                   model,
                                                   st_seed,
                                                   test_num=test_num,
                                                   seed_list=seed_list,
                                                   video_size=video_size,
                                                   instruction_type=instruction_type,
                                                   expert_check=bool(usr_args.get("expert_check", True)),
                                                   initial_success_count=initial_success_count,
                                                   initial_test_count=initial_test_count)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        denom = max(1, int(final_test_num))
        file.write("\n".join(map(str, np.array(suc_nums) / denom)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                seed_list=None,
                video_size=None,
                instruction_type=None,
                expert_check=True,
                initial_success_count=0,
                initial_test_count=0):
    print(colorize(f"Task Name: {args['task_name']}", "34"))
    print(colorize(f"Policy Name: {args['policy_name']}", "34"))
    print(colorize(f"Expert Check: {expert_check}", "34"))
    TASK_ENV.suc = int(initial_success_count)
    TASK_ENV.test_num = int(initial_test_count)

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    use_seed_list = seed_list is not None
    target_eval_num = len(seed_list) if use_seed_list else test_num
    if use_seed_list and target_eval_num > 0:
        now_seed = seed_list[0]

    while (now_id < target_eval_num) if use_seed_list else (succ_seed < test_num):
        if use_seed_list:
            now_seed = seed_list[now_id]

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        episode_info = None
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                if use_seed_list:
                    print(f"skip unstable seed: {now_seed}")
                    now_id += 1
                else:
                    now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                if use_seed_list:
                    print(f"skip seed due to exception: {now_seed}, error: {e}")
                    now_id += 1
                else:
                    now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            if use_seed_list:
                print(f"skip seed due to expert check failed: {now_seed}")
                now_id += 1
            else:
                now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        if episode_info is not None and isinstance(episode_info, dict):
            episode_meta = episode_info.get("info", {})
        elif hasattr(TASK_ENV, "info") and isinstance(TASK_ENV.info, dict):
            episode_meta = TASK_ENV.info.get("info", {})
        else:
            episode_meta = {}
        episode_info_list = [episode_meta]
        random.seed(now_seed)
        np.random.seed(now_seed)
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction_candidates = []
        if results:
            instruction_candidates = results[0].get(instruction_type, []) or []
        instruction = np.random.choice(instruction_candidates) if instruction_candidates else None
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print(colorize("Success!", "92"))
        else:
            print(colorize("Fail!", "91"))

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        success_ratio = f"{TASK_ENV.suc}/{TASK_ENV.test_num}"
        success_pct = f"{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%"
        print(
            f"{colorize(task_name, '93')} | {colorize(args['policy_name'], '94')} | {colorize(args['task_config'], '92')} | {colorize(args['ckpt_setting'], '91')}\n"
            f"Success rate: {colorize(success_ratio, '96')} => {colorize(success_pct, '95')}, current seed: {colorize(str(now_seed), '90')}\n"
        )
        # TASK_ENV._take_picture()
        if not use_seed_list:
            now_seed += 1

    return now_seed, TASK_ENV.suc, TASK_ENV.test_num


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            if isinstance(value, str):
                lower = value.strip().lower()
                if lower == "true":
                    value = True
                elif lower == "false":
                    value = False
                elif lower == "none":
                    value = None
                else:
                    try:
                        value = eval(value)
                    except:
                        pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
