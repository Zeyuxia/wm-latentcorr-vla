from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4,)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz").astype(np.float32)


def _poses_to_cosmos_state_data(
    fk_poses: np.ndarray,
    grippers: np.ndarray,
) -> dict[str, list]:
    fk_poses = np.asarray(fk_poses, dtype=np.float32)
    grippers = np.asarray(grippers, dtype=np.float32)
    if fk_poses.ndim != 2 or fk_poses.shape[1] != 14:
        raise ValueError(f"Expected fk_poses shape [T,14], got {fk_poses.shape}")
    if grippers.ndim != 2 or grippers.shape[1] != 2 or grippers.shape[0] != fk_poses.shape[0]:
        raise ValueError(f"Expected grippers shape [T,2] aligned with poses, got {grippers.shape}")

    left_states = []
    right_states = []
    for pose in fk_poses:
        lp = np.asarray(pose[0:3], dtype=np.float32)
        lq = np.asarray(pose[3:7], dtype=np.float32)
        rp = np.asarray(pose[7:10], dtype=np.float32)
        rq = np.asarray(pose[10:14], dtype=np.float32)
        left_states.append(np.concatenate([lp, _quat_wxyz_to_euler_xyz(lq)], axis=0).astype(float).tolist())
        right_states.append(np.concatenate([rp, _quat_wxyz_to_euler_xyz(rq)], axis=0).astype(float).tolist())

    return {
        "state": left_states,
        "continuous_gripper_state": grippers[:, 0].astype(float).tolist(),
        "state_right": right_states,
        "continuous_gripper_state_right": grippers[:, 1].astype(float).tolist(),
    }


class CosmosActionConditionedRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        cosmos_root = Path(args.cosmos_root).resolve()
        config_file = str(args.config_file)
        if Path(config_file).is_absolute():
            try:
                config_file = str(Path(config_file).resolve().relative_to(cosmos_root))
            except ValueError:
                pass
        # Cosmos treats config_file as an import-style path. Passing an absolute
        # path under a repo with hyphens makes it try to import ".data....".
        args.config_file = config_file
        if str(cosmos_root) not in sys.path:
            sys.path.insert(0, str(cosmos_root))

        old_cwd = os.getcwd()
        try:
            # Some Cosmos config imports assume the repository root as cwd.
            # Keep that behavior during initialization, but do not leak it to
            # the SmolVLA process when this runner is used in direct mode.
            os.chdir(cosmos_root)
            import torchvision
            from cosmos_predict2._src.predict2.action.robot_action_utils import get_action_sequence_from_states
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
            from cosmos_predict2.config import DEFAULT_NEGATIVE_PROMPT

            self.torchvision = torchvision
            self.get_action_sequence_from_states = get_action_sequence_from_states
            self.default_negative_prompt = DEFAULT_NEGATIVE_PROMPT
            self.video2world = Video2WorldInference(
                experiment_name=args.experiment,
                ckpt_path=args.checkpoint_path,
                s3_credential_path="",
                context_parallel_size=int(args.context_parallel_size),
                config_file=args.config_file,
            )
        finally:
            os.chdir(old_cwd)
        self.expected_action_dim = int(self.video2world.model.config.net.action_dim)

    def _build_actions(self, fk_poses: np.ndarray, grippers: np.ndarray) -> np.ndarray:
        data = _poses_to_cosmos_state_data(fk_poses, grippers)
        actions = self.get_action_sequence_from_states(
            data,
            fps_downsample_ratio=int(self.args.fps_downsample_ratio),
            state_key="state",
            gripper_scale=float(self.args.gripper_scale),
            gripper_key="continuous_gripper_state",
            action_scaler=float(self.args.action_scaler),
            use_quat=bool(self.args.use_quat),
            right_state_key="state_right",
            right_gripper_key="continuous_gripper_state_right",
            invert_gripper=bool(self.args.invert_gripper),
            action_stats_path=str(self.args.action_stats_path) if str(self.args.action_stats_path).strip() else None,
            action_normalization_clip=_parse_optional_float(self.args.action_normalization_clip),
        )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.expected_action_dim:
            raise ValueError(
                f"Cosmos action shape {actions.shape} does not match model action_dim={self.expected_action_dim}"
            )
        return actions

    def _run_impl(
        self,
        curr_image: np.ndarray,
        fk_poses: np.ndarray,
        grippers: np.ndarray,
        *,
        save_dir: str | None = None,
    ) -> dict:
        import cv2
        import mediapy

        if curr_image.ndim != 3 or curr_image.shape[0] != 3:
            raise ValueError(f"Expected curr_image shape [3,H,W], got {curr_image.shape}")

        img_array = np.clip(curr_image.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        target_h, target_w = [int(x) for x in str(self.args.resolution).split(",")]
        if tuple(img_array.shape[:2]) != (target_h, target_w):
            img_array = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        actions = self._build_actions(fk_poses, grippers)
        if actions.shape[0] <= 0:
            raise ValueError("Cosmos needs at least one relative action; got an empty action sequence.")

        frames_out = [img_array]
        chunk_size = int(self.args.chunk_size)
        chunk_records = []
        prompt = str(self.args.prompt or "")
        negative_prompt = str(self.args.negative_prompt or "") or self.default_negative_prompt

        with torch.no_grad():
            for local_action_idx in range(0, int(actions.shape[0]), chunk_size):
                actions_valid = actions[local_action_idx : local_action_idx + chunk_size]
                valid_len = int(actions_valid.shape[0])
                actions_chunk = actions_valid
                if valid_len < chunk_size:
                    pad = np.zeros((chunk_size - valid_len, actions.shape[1]), dtype=actions.dtype)
                    actions_chunk = np.concatenate([actions_valid, pad], axis=0)

                img_tensor = self.torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)
                num_video_frames = int(actions_chunk.shape[0]) + 1
                vid_input = torch.cat(
                    [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
                    dim=0,
                )
                vid_input = (vid_input * 255.0).to(torch.uint8)
                vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

                video = self.video2world.generate_vid2world(
                    prompt=prompt,
                    input_path=vid_input,
                    action=torch.from_numpy(actions_chunk).float(),
                    guidance=int(self.args.guidance),
                    num_video_frames=num_video_frames,
                    num_latent_conditional_frames=int(self.args.num_latent_conditional_frames),
                    resolution=str(self.args.resolution),
                    seed=int(self.args.seed) + int(local_action_idx),
                    negative_prompt=negative_prompt,
                    num_steps=int(self.args.num_steps),
                    fps=float(self.args.save_fps),
                )
                video_normalized = (video + 1.0) / 2.0
                video_clamped = (
                    (torch.clamp(video_normalized[0], 0, 1) * 255)
                    .to(torch.uint8)
                    .permute(1, 2, 3, 0)
                    .cpu()
                    .numpy()
                )
                img_array = video_clamped[valid_len]
                frames_out.extend([frame for frame in video_clamped[1 : valid_len + 1]])
                chunk_records.append(
                    {
                        "local_action_idx": int(local_action_idx),
                        "valid_len": int(valid_len),
                        "padded_len": int(chunk_size - valid_len),
                    }
                )

        out_img = img_array.astype(np.float32).transpose(2, 0, 1) / 255.0
        meta = {
            "backend": "cosmos",
            "checkpoint_path": str(self.args.checkpoint_path),
            "experiment": str(self.args.experiment),
            "config_file": str(self.args.config_file),
            "resolution": str(self.args.resolution),
            "chunk_size": int(chunk_size),
            "num_actions": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "fps_downsample_ratio": int(self.args.fps_downsample_ratio),
            "invert_gripper": bool(self.args.invert_gripper),
            "action_scaler": float(self.args.action_scaler),
            "action_stats_path": str(self.args.action_stats_path),
            "action_normalization_clip": _parse_optional_float(self.args.action_normalization_clip),
            "num_steps": int(self.args.num_steps),
            "guidance": int(self.args.guidance),
            "save_fps": int(self.args.save_fps),
            "chunks": chunk_records,
        }
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            video_path = os.path.join(save_dir, "outputs.mp4")
            mediapy.write_video(video_path, np.asarray(frames_out, dtype=np.uint8), fps=int(self.args.save_fps))
            cv2.imwrite(os.path.join(save_dir, "input_frame.png"), frames_out[0][:, :, ::-1])
            with open(os.path.join(save_dir, "cosmos_runtime_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        return {"out_img": out_img, "meta": meta}

    def run_request(self, request: dict) -> dict:
        request_npz = str(request["request_npz"])
        response_npy = str(request["response_npy"])
        save_dir = request.get("save_dir")

        payload = np.load(request_npz)
        curr_image = np.asarray(payload["curr_image"], dtype=np.float32)
        fk_poses = np.asarray(payload["fk_poses"], dtype=np.float32)
        grippers = np.asarray(payload["grippers"], dtype=np.float32)
        result = self._run_impl(curr_image, fk_poses, grippers, save_dir=save_dir)
        out_img = result["out_img"]
        np.save(response_npy, out_img)
        return {
            "ok": True,
            "response_npy": response_npy,
            "meta": result["meta"],
        }

    def run_batch(
        self,
        curr_image: torch.Tensor,
        fk_poses: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        grippers: list[tuple[float, float]],
        *,
        save_dir: str | None = None,
    ) -> torch.Tensor:
        curr_image_np = torch.clamp(curr_image.detach().float().cpu(), 0.0, 1.0).numpy().astype(np.float32)
        fk_poses_np = np.asarray(
            [
                np.concatenate(
                    [
                        np.asarray(lp, dtype=np.float32).reshape(3,),
                        np.asarray(lq, dtype=np.float32).reshape(4,),
                        np.asarray(rp, dtype=np.float32).reshape(3,),
                        np.asarray(rq, dtype=np.float32).reshape(4,),
                    ],
                    axis=0,
                )
                for (lp, lq, rp, rq) in fk_poses
            ],
            dtype=np.float32,
        )
        grippers_np = np.asarray(grippers, dtype=np.float32)
        result = self._run_impl(curr_image_np, fk_poses_np, grippers_np, save_dir=save_dir)
        return torch.from_numpy(result["out_img"]).float()

    def close(self) -> None:
        try:
            self.video2world.cleanup()
        except Exception:
            pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cosmos action-conditioned rollout worker")
    parser.add_argument("--socket_path", type=str, required=True)
    parser.add_argument("--cosmos_root", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=12)
    parser.add_argument("--guidance", type=int, default=7)
    parser.add_argument("--resolution", type=str, default="256,320")
    parser.add_argument("--fps_downsample_ratio", type=int, default=1)
    parser.add_argument("--gripper_scale", type=float, default=1.0)
    parser.add_argument("--invert_gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--save_fps", type=int, default=30)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=1)
    parser.add_argument("--action_scaler", type=float, default=20.0)
    parser.add_argument("--action_stats_path", type=str, default="")
    parser.add_argument("--action_normalization_clip", type=str, default="")
    parser.add_argument("--use_quat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    runner = CosmosActionConditionedRunner(args)
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=b"smolvla_cosmos")
    try:
        while True:
            conn = listener.accept()
            try:
                request = conn.recv()
                if isinstance(request, dict) and request.get("op") == "shutdown":
                    conn.send({"ok": True, "shutdown": True})
                    break
                response = runner.run_request(request)
                conn.send(response)
            except Exception as exc:
                conn.send(
                    {
                        "ok": False,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                conn.close()
    finally:
        listener.close()
        runner.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
