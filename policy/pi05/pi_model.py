#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, camera_names=None):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.camera_names = tuple(camera_names or ("cam_high", "cam_right_wrist", "cam_left_wrist"))

        specified_path = f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        if len(img_arr) != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} images, got {len(img_arr)}")

        images = {}
        for camera_name, image in zip(self.camera_names, img_arr):
            images[camera_name] = np.transpose(image, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": images,
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
