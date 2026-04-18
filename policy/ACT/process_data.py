import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
RAW_TO_PROCESSED_CAMERA_NAMES = {
    "head_camera": "cam_high",
    "right_camera": "cam_right_wrist",
    "left_camera": "cam_left_wrist",
}


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def data_transform(path, episode_num, save_path, camera_names):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        qpos = []
        actions = []
        image_buffers = {cam_name: [] for cam_name in camera_names}
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            if j != left_gripper_all.shape[0] - 1:
                state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint

                state = state.astype(np.float32)
                qpos.append(state)

                for raw_cam_name, processed_cam_name in RAW_TO_PROCESSED_CAMERA_NAMES.items():
                    if processed_cam_name not in image_buffers:
                        continue
                    cam_bits = image_dict[raw_cam_name][j]
                    cam_image = cv2.imdecode(np.frombuffer(cam_bits, np.uint8), cv2.IMREAD_COLOR)
                    cam_image_resized = cv2.resize(cam_image, (640, 480))
                    image_buffers[processed_cam_name].append(cam_image_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            for cam_name in camera_names:
                image.create_dataset(cam_name, data=np.stack(image_buffers[cam_name]), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument(
        "--camera_names",
        type=str,
        default="cam_high,cam_right_wrist,cam_left_wrist",
        help="Comma-separated processed camera names to keep, e.g. cam_high",
    )

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num
    camera_names = [x.strip() for x in args.camera_names.split(",") if x.strip()]
    valid_camera_names = set(RAW_TO_PROCESSED_CAMERA_NAMES.values())
    invalid_camera_names = [x for x in camera_names if x not in valid_camera_names]
    if invalid_camera_names:
        raise ValueError(f"Unsupported camera_names: {invalid_camera_names}, valid={sorted(valid_camera_names)}")
    if not camera_names:
        raise ValueError("camera_names cannot be empty")

    begin = 0
    raw_data_dir = os.path.join(REPO_ROOT, "data", task_name, task_config, "data")
    save_path = os.path.join(CURRENT_DIR, "processed_data", f"sim-{task_name}", f"{task_config}-{expert_data_num}")
    begin = data_transform(
        raw_data_dir,
        expert_data_num,
        save_path,
        camera_names,
    )

    SIM_TASK_CONFIGS_PATH = os.path.join(CURRENT_DIR, "SIM_TASK_CONFIGS.json")

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": camera_names,
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
