import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


def _extract_rgb(camera_source, key):
    value = camera_source[key]
    if isinstance(value, dict) and "rgb" in value:
        return value["rgb"]
    return value


def _camera_names_from_args(usr_args):
    camera_mode = str(usr_args.get("camera_mode", "head_only")).strip().lower()
    secondary_camera = str(usr_args.get("secondary_camera", "right_wrist")).strip().lower()
    if camera_mode == "head_only":
        return ("cam_high",), ("head_camera",)
    if camera_mode == "dual_view":
        if secondary_camera == "left_wrist":
            return ("cam_high", "cam_left_wrist"), ("head_camera", "left_camera")
        if secondary_camera == "right_wrist":
            return ("cam_high", "cam_right_wrist"), ("head_camera", "right_camera")
        raise ValueError(f"Unsupported secondary_camera={secondary_camera}")
    if camera_mode == "tri_view":
        return ("cam_high", "cam_left_wrist", "cam_right_wrist"), ("head_camera", "left_camera", "right_camera")
    raise ValueError(f"Unsupported camera_mode={camera_mode}")


# Encode observation for the model
def encode_obs(observation, runtime_camera_keys):
    camera_source = observation["observation"]
    input_rgb_arr = [_extract_rgb(camera_source, key) for key in runtime_camera_keys]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    camera_names, runtime_camera_keys = _camera_names_from_args(usr_args)
    model = PI0(
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        camera_names=camera_names,
    )
    model.runtime_camera_keys = runtime_camera_keys
    return model


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation, getattr(model, "runtime_camera_keys", ("head_camera",)))
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation, getattr(model, "runtime_camera_keys", ("head_camera",)))
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
