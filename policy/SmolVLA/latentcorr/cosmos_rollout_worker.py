from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
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


@contextmanager
def _cosmos_log_context():
    if str(os.environ.get("SMOLVLA_COSMOS_VERBOSE", "")).strip().lower() in {"1", "true", "yes", "on"}:
        with nullcontext():
            yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


@contextmanager
def _single_rank_distributed_context(enabled: bool = True):
    """Make Cosmos behave like a rank-local single-GPU world model.

    SmolVLA already runs under accelerate/DDP.  For EVAC-style compare, each
    rank should load and run its own world model on its own CUDA device.  Cosmos'
    loader otherwise notices the outer process group and tries to do rank0-only
    checkpoint loading plus broadcasts across all SmolVLA ranks.  Patch only
    Cosmos' distributed helper module so the outer torch.distributed group stays
    intact for SmolVLA progress synchronization.
    """
    if not enabled:
        yield
        return

    patched = []
    env_keys = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
    )
    old_env = {key: os.environ.get(key) for key in env_keys}

    def _patch(obj, name: str, value) -> None:
        if hasattr(obj, name):
            patched.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

    try:
        # Cosmos' inference stack was written to also support torchrun/context
        # parallelism.  SmolVLA already owns the outer process group via
        # accelerate, so make Cosmos see a rank-local single-process world while
        # it initializes and runs.  Do not destroy or modify the real outer
        # process group.
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["GROUP_RANK"] = "0"
        os.environ["ROLE_RANK"] = "0"
        os.environ["ROLE_WORLD_SIZE"] = "1"

        from cosmos_predict2._src.imaginaire.utils import distributed as cosmos_distributed

        _patch(cosmos_distributed, "init", lambda: 0)
        _patch(cosmos_distributed, "get_rank", lambda group=None: 0)
        _patch(cosmos_distributed, "get_world_size", lambda group=None: 1)
        _patch(cosmos_distributed, "is_rank0", lambda: True)
        _patch(cosmos_distributed, "is_local_rank0", lambda: True)
        _patch(cosmos_distributed, "barrier", lambda: None)
        _patch(cosmos_distributed, "broadcast", lambda tensor, *args, **kwargs: tensor)
        _patch(cosmos_distributed, "sync_model_states", lambda *args, **kwargs: None)
        _patch(cosmos_distributed, "all_gather_tensor", lambda tensor: [tensor])
        _patch(cosmos_distributed, "gather_object", lambda payload: [payload])
        _patch(cosmos_distributed, "dist_reduce_tensor", lambda tensor, *args, **kwargs: tensor)
    except Exception:
        pass
    try:
        from cosmos_predict2._src.imaginaire.utils import log as cosmos_log

        _patch(cosmos_log, "_get_rank", lambda group=None: 0)
    except Exception:
        pass
    try:
        import cosmos_predict2.config as cosmos_config

        _patch(cosmos_config, "is_rank0", lambda: True)
        try:
            cosmos_config.is_rank0.cache_clear()
        except Exception:
            pass
    except Exception:
        pass

    try:
        yield
    finally:
        for obj, name, old_value in reversed(patched):
            setattr(obj, name, old_value)
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4,)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz").astype(np.float32)


def _quat_xyzw_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4,)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    return R.from_quat(q).as_euler("xyz").astype(np.float32)


def _quat_to_cosmos_euler_xyz(quat: np.ndarray, quat_input_order: str) -> np.ndarray:
    order = str(quat_input_order or "xyzw").strip().lower()
    if order == "wxyz":
        return _quat_wxyz_to_euler_xyz(quat)
    if order == "xyzw":
        return _quat_xyzw_to_euler_xyz(quat)
    raise ValueError(f"Unsupported Cosmos quat_input_order={quat_input_order!r}; expected 'wxyz' or 'xyzw'.")


def _poses_to_cosmos_state_data(
    fk_poses: np.ndarray,
    grippers: np.ndarray,
    *,
    quat_input_order: str = "wxyz",
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
        left_states.append(
            np.concatenate([lp, _quat_to_cosmos_euler_xyz(lq, quat_input_order)], axis=0).astype(float).tolist()
        )
        right_states.append(
            np.concatenate([rp, _quat_to_cosmos_euler_xyz(rq, quat_input_order)], axis=0).astype(float).tolist()
        )

    return {
        "state": left_states,
        "continuous_gripper_state": grippers[:, 0].astype(float).tolist(),
        "state_right": right_states,
        "continuous_gripper_state_right": grippers[:, 1].astype(float).tolist(),
    }


class CosmosActionConditionedRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._single_rank_distributed = int(getattr(args, "context_parallel_size", 1)) <= 1
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
            with _single_rank_distributed_context(self._single_rank_distributed):
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
        data = _poses_to_cosmos_state_data(
            fk_poses,
            grippers,
            quat_input_order=str(getattr(self.args, "quat_input_order", "xyzw")),
        )
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

    @staticmethod
    def _action_summary(actions: np.ndarray) -> dict:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.size == 0:
            return {"shape": list(arr.shape), "empty": True}
        left = arr[:, :7]
        right = arr[:, 7:14] if arr.shape[1] >= 14 else np.zeros((arr.shape[0], 0), dtype=arr.dtype)
        payload = {
            "shape": [int(x) for x in arr.shape],
            "empty": False,
            "min": arr.min(axis=0).astype(float).tolist(),
            "max": arr.max(axis=0).astype(float).tolist(),
            "mean": arr.mean(axis=0).astype(float).tolist(),
            "first": arr[0].astype(float).tolist(),
            "last": arr[-1].astype(float).tolist(),
        }
        if left.size:
            payload["left_xyz_sum"] = left[:, :3].sum(axis=0).astype(float).tolist()
            payload["left_xyz_l2_total"] = float(np.linalg.norm(left[:, :3], axis=1).sum())
        if right.size:
            payload["right_xyz_sum"] = right[:, :3].sum(axis=0).astype(float).tolist()
            payload["right_xyz_l2_total"] = float(np.linalg.norm(right[:, :3], axis=1).sum())
        return payload

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

        with torch.no_grad(), _single_rank_distributed_context(self._single_rank_distributed):
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

                with _cosmos_log_context():
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
            "quat_input_order": str(getattr(self.args, "quat_input_order", "xyzw")),
            "action_scaler": float(self.args.action_scaler),
            "action_stats_path": str(self.args.action_stats_path),
            "action_normalization_clip": _parse_optional_float(self.args.action_normalization_clip),
            "num_steps": int(self.args.num_steps),
            "guidance": int(self.args.guidance),
            "save_fps": int(self.args.save_fps),
            "action_summary": self._action_summary(actions),
            "chunks": chunk_records,
        }
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            video_path = os.path.join(save_dir, "outputs.mp4")
            mediapy.write_video(video_path, np.asarray(frames_out, dtype=np.uint8), fps=int(self.args.save_fps))
            cv2.imwrite(os.path.join(save_dir, "input_frame.png"), frames_out[0][:, :, ::-1])
            np.save(os.path.join(save_dir, "actions.npy"), actions.astype(np.float32))
            with open(os.path.join(save_dir, "action_summary.json"), "w", encoding="utf-8") as f:
                json.dump(meta["action_summary"], f, indent=2, ensure_ascii=False)
            with open(os.path.join(save_dir, "cosmos_runtime_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        return {"out_img": out_img, "meta": meta}

    def run_request(self, request: dict) -> dict:
        request_npz = str(request["request_npz"])
        response_npy = str(request["response_npy"])
        save_dir = request.get("save_dir")

        try:
            with np.load(request_npz) as payload:
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
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

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
        try:
            result = self._run_impl(curr_image_np, fk_poses_np, grippers_np, save_dir=save_dir)
            return torch.from_numpy(result["out_img"]).float()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

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
    parser.add_argument("--quat_input_order", type=str, choices=["wxyz", "xyzw"], default="wxyz")
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
