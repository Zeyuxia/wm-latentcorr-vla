from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import tempfile
import time
import traceback
import uuid
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CosmosRolloutClient:
    def __init__(
        self,
        *,
        cosmos_root: str,
        python_bin: str,
        checkpoint_path: str,
        experiment: str,
        config_file: str,
        device: str = "cuda",
        cuda_visible_devices: str | None = None,
        context_parallel_size: int = 1,
        chunk_size: int = 12,
        guidance: int = 7,
        resolution: str = "256,320",
        fps_downsample_ratio: int = 1,
        gripper_scale: float = 1.0,
        invert_gripper: bool = True,
        num_steps: int = 35,
        save_fps: int = 30,
        num_latent_conditional_frames: int = 1,
        action_scaler: float = 20.0,
        action_stats_path: str = "",
        action_normalization_clip: float | None = None,
        use_quat: bool = False,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 0,
        work_dir: str | None = None,
        execution_mode: str = "worker",
        startup_timeout_s: float = 600.0,
        request_timeout_s: float = 900.0,
    ):
        execution_mode = str(execution_mode or "worker").strip().lower()
        if execution_mode not in {"worker", "direct"}:
            raise ValueError(f"Unsupported Cosmos execution_mode={execution_mode!r}; expected 'worker' or 'direct'.")
        self.cosmos_root = str(Path(cosmos_root).resolve())
        self.python_bin = str(python_bin)
        self.checkpoint_path = str(checkpoint_path)
        self.experiment = str(experiment)
        self.config_file = str(config_file)
        self.device = str(device)
        self.cuda_visible_devices = None if cuda_visible_devices is None else str(cuda_visible_devices)
        self.context_parallel_size = int(context_parallel_size)
        self.chunk_size = int(chunk_size)
        self.guidance = int(guidance)
        self.resolution = str(resolution)
        self.fps_downsample_ratio = int(fps_downsample_ratio)
        self.gripper_scale = float(gripper_scale)
        self.invert_gripper = bool(invert_gripper)
        self.num_steps = int(num_steps)
        self.save_fps = int(save_fps)
        self.num_latent_conditional_frames = int(num_latent_conditional_frames)
        self.action_scaler = float(action_scaler)
        self.action_stats_path = str(action_stats_path or "")
        self.action_normalization_clip = action_normalization_clip
        self.use_quat = bool(use_quat)
        self.prompt = str(prompt or "")
        self.negative_prompt = str(negative_prompt or "")
        self.seed = int(seed)
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.execution_mode = execution_mode
        explicit_work_dir = str(work_dir or "").strip()
        if explicit_work_dir:
            self.work_dir = explicit_work_dir
            os.makedirs(self.work_dir, exist_ok=True)
        elif execution_mode == "worker":
            self.work_dir = tempfile.mkdtemp(prefix="smolvla_cosmos_")
        else:
            self.work_dir = ""

        if execution_mode == "worker":
            # AF_UNIX socket paths are capped at ~108 bytes on Linux. Explore
            # output dirs are intentionally descriptive and can easily exceed
            # that, so keep the IPC socket in /tmp while logs/artifacts stay in
            # work_dir.
            socket_dir = os.path.join(tempfile.gettempdir(), "smolvla_cosmos_sockets")
            os.makedirs(socket_dir, exist_ok=True)
            self.socket_path = os.path.join(socket_dir, f"cw_{uuid.uuid4().hex[:24]}.sock")
        else:
            self.socket_path = ""
        log_name = "cosmos_direct.log" if execution_mode == "direct" else "cosmos_worker.log"
        self.log_path = os.path.join(self.work_dir, log_name) if self.work_dir else ""
        self.proc: subprocess.Popen | None = None
        self.runner: Any | None = None
        atexit.register(self.close)

    @property
    def runtime_meta_filename(self) -> str:
        return "cosmos_runtime_meta.json"

    def _worker_script(self) -> str:
        return str(Path(__file__).resolve().with_name("cosmos_rollout_worker.py"))

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.python_bin,
            self._worker_script(),
            "--socket_path",
            self.socket_path,
            "--cosmos_root",
            self.cosmos_root,
            "--checkpoint_path",
            self.checkpoint_path,
            "--experiment",
            self.experiment,
            "--config_file",
            self.config_file,
            "--context_parallel_size",
            str(self.context_parallel_size),
            "--chunk_size",
            str(self.chunk_size),
            "--guidance",
            str(self.guidance),
            "--resolution",
            self.resolution,
            "--fps_downsample_ratio",
            str(self.fps_downsample_ratio),
            "--gripper_scale",
            str(self.gripper_scale),
            "--num_steps",
            str(self.num_steps),
            "--save_fps",
            str(self.save_fps),
            "--num_latent_conditional_frames",
            str(self.num_latent_conditional_frames),
            "--action_scaler",
            str(self.action_scaler),
            "--action_stats_path",
            self.action_stats_path,
            "--action_normalization_clip",
            "" if self.action_normalization_clip is None else str(float(self.action_normalization_clip)),
            "--prompt",
            self.prompt,
            "--negative_prompt",
            self.negative_prompt,
            "--seed",
            str(self.seed),
        ]
        cmd.append("--invert_gripper" if self.invert_gripper else "--no-invert_gripper")
        cmd.append("--use_quat" if self.use_quat else "--no-use_quat")
        return cmd

    def _build_runner_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            socket_path=self.socket_path,
            cosmos_root=self.cosmos_root,
            checkpoint_path=self.checkpoint_path,
            experiment=self.experiment,
            config_file=self.config_file,
            context_parallel_size=self.context_parallel_size,
            chunk_size=self.chunk_size,
            guidance=self.guidance,
            resolution=self.resolution,
            fps_downsample_ratio=self.fps_downsample_ratio,
            gripper_scale=self.gripper_scale,
            invert_gripper=self.invert_gripper,
            num_steps=self.num_steps,
            save_fps=self.save_fps,
            num_latent_conditional_frames=self.num_latent_conditional_frames,
            action_scaler=self.action_scaler,
            action_stats_path=self.action_stats_path,
            action_normalization_clip=(
                "" if self.action_normalization_clip is None else str(float(self.action_normalization_clip))
            ),
            use_quat=self.use_quat,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
        )

    def _write_error(
        self,
        payload: dict[str, Any] | None,
        exc: BaseException,
        *,
        traceback_text: str | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        tb = traceback_text if traceback_text is not None else traceback.format_exc()
        error_payload: dict[str, Any] = {
            "backend": "cosmos",
            "execution_mode": self.execution_mode,
            "error": repr(exc),
            "traceback": tb,
            "cosmos_root": self.cosmos_root,
            "checkpoint_path": self.checkpoint_path,
            "experiment": self.experiment,
            "config_file": self.config_file,
            "worker_log_path": self.log_path,
        }
        if isinstance(payload, dict):
            error_payload["request_npz"] = str(payload.get("request_npz", ""))
            error_payload["response_npy"] = str(payload.get("response_npy", ""))
        if isinstance(response, dict):
            error_payload["worker_response"] = response

        save_dir = payload.get("save_dir") if isinstance(payload, dict) else None
        if save_dir:
            try:
                os.makedirs(str(save_dir), exist_ok=True)
                with open(os.path.join(str(save_dir), "cosmos_error.json"), "w", encoding="utf-8") as f:
                    json.dump(error_payload, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        try:
            if self.work_dir:
                os.makedirs(self.work_dir, exist_ok=True)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write("\n[CosmosRolloutClient] ERROR\n")
                    f.write(json.dumps(error_payload, ensure_ascii=False, indent=2))
                    f.write("\n")
        except Exception:
            pass

    def start(self) -> None:
        if self.execution_mode == "direct":
            if self.runner is not None:
                return
            env = os.environ
            if self.work_dir:
                env.setdefault("MPLCONFIGDIR", os.path.join(self.work_dir, "mplconfig"))
            env.setdefault("HF_HOME", "/data/zhenyangfan/.cache/huggingface")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
            env.setdefault("HF_HUB_OFFLINE", "1")
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.set_device(torch.device(self.device))
            from policy.SmolVLA.latentcorr.cosmos_rollout_worker import CosmosActionConditionedRunner

            if self.work_dir:
                os.makedirs(self.work_dir, exist_ok=True)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write("\n[CosmosRolloutClient] Starting direct runner\n")
                    f.write(f"device={self.device}\n")
                    f.write(f"cosmos_root={self.cosmos_root}\n")
                    f.write(f"checkpoint_path={self.checkpoint_path}\n")
                    f.flush()
            self.runner = CosmosActionConditionedRunner(self._build_runner_args())
            return

        if self.proc is not None and self.proc.poll() is None and os.path.exists(self.socket_path):
            return
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = self.cosmos_root
        env.setdefault("MPLCONFIGDIR", os.path.join(self.work_dir, "mplconfig"))
        env.setdefault("HF_HOME", "/data/zhenyangfan/.cache/huggingface")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        stdout = open(self.log_path, "a", encoding="utf-8")
        stdout.write("\n[CosmosRolloutClient] Starting worker\n")
        stdout.write(" ".join(self._build_cmd()) + "\n")
        stdout.flush()
        self.proc = subprocess.Popen(
            self._build_cmd(),
            cwd=self.cosmos_root,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Cosmos worker exited with code {self.proc.returncode}. See log: {self.log_path}"
                )
            if os.path.exists(self.socket_path):
                return
            time.sleep(1.0)
        raise TimeoutError(f"Timed out waiting for Cosmos worker socket: {self.socket_path}. Log: {self.log_path}")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.execution_mode == "direct":
            try:
                self.start()
                if self.runner is None:
                    raise RuntimeError("Cosmos direct runner failed to initialize.")
                response = self.runner.run_request(payload)
                if not isinstance(response, dict):
                    raise RuntimeError(f"Unexpected Cosmos direct response: {type(response)!r}")
                if not bool(response.get("ok", False)):
                    raise RuntimeError(f"Cosmos direct runner failed: {response}")
                return response
            except Exception as exc:
                self._write_error(payload, exc)
                raise

        try:
            self.start()
        except Exception as exc:
            self._write_error(payload, exc)
            raise
        deadline = time.time() + self.request_timeout_s
        last_exc: Exception | None = None
        while time.time() < deadline:
            try:
                conn = Client(str(self.socket_path), family="AF_UNIX", authkey=b"smolvla_cosmos")
                try:
                    conn.send(payload)
                    response = conn.recv()
                finally:
                    conn.close()
                if not isinstance(response, dict):
                    exc = RuntimeError(f"Unexpected Cosmos worker response: {type(response)!r}")
                    self._write_error(payload, exc)
                    raise exc
                if not bool(response.get("ok", False)):
                    exc = RuntimeError(
                        f"Cosmos worker failed: {response.get('error')}\n{response.get('traceback', '')}"
                    )
                    self._write_error(
                        payload,
                        exc,
                        traceback_text=str(response.get("traceback", "")),
                        response=response,
                    )
                    raise exc
                return response
            except (ConnectionRefusedError, FileNotFoundError, EOFError) as exc:
                last_exc = exc
                if self.proc is not None and self.proc.poll() is not None:
                    runtime_exc = RuntimeError(
                        f"Cosmos worker exited with code {self.proc.returncode}. See log: {self.log_path}"
                    )
                    self._write_error(payload, runtime_exc)
                    raise runtime_exc from exc
                time.sleep(0.5)
        exc = TimeoutError(f"Timed out waiting for Cosmos response; last error={last_exc!r}. Log: {self.log_path}")
        self._write_error(payload, exc)
        raise exc

    def infer_batch(
        self,
        curr_image: torch.Tensor,
        fk_poses: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        grippers: list[tuple[float, float]],
        *,
        save_dir: str | None = None,
    ) -> torch.Tensor:
        if self.execution_mode != "direct":
            return self.infer(curr_image, fk_poses, grippers, save_dir=save_dir)
        try:
            self.start()
            if self.runner is None:
                raise RuntimeError("Cosmos direct runner failed to initialize.")
            response = self.runner.run_batch(curr_image, fk_poses, grippers, save_dir=save_dir)
            if not isinstance(response, torch.Tensor):
                raise RuntimeError(f"Unexpected Cosmos direct batch response: {type(response)!r}")
            return response.float()
        except Exception as exc:
            self._write_error(None, exc)
            raise

    def infer(
        self,
        curr_image: torch.Tensor,
        fk_poses: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        grippers: list[tuple[float, float]],
        *,
        save_dir: str | None = None,
    ) -> torch.Tensor:
        if self.execution_mode == "direct":
            return self.infer_batch(curr_image, fk_poses, grippers, save_dir=save_dir)
        request_id = uuid.uuid4().hex
        request_npz = os.path.join(self.work_dir, f"request_{request_id}.npz")
        response_npy = os.path.join(self.work_dir, f"response_{request_id}.npy")
        pose_arr = np.zeros((len(fk_poses), 14), dtype=np.float32)
        grip_arr = np.zeros((len(grippers), 2), dtype=np.float32)
        for idx, ((lp, lq, rp, rq), (lg, rg)) in enumerate(zip(fk_poses, grippers)):
            pose_arr[idx, 0:3] = np.asarray(lp, dtype=np.float32).reshape(3,)
            pose_arr[idx, 3:7] = np.asarray(lq, dtype=np.float32).reshape(4,)
            pose_arr[idx, 7:10] = np.asarray(rp, dtype=np.float32).reshape(3,)
            pose_arr[idx, 10:14] = np.asarray(rq, dtype=np.float32).reshape(4,)
            grip_arr[idx, :] = [float(lg), float(rg)]
        img = torch.clamp(curr_image.detach().float().cpu(), 0.0, 1.0).numpy().astype(np.float32)
        np.savez_compressed(
            request_npz,
            curr_image=img,
            fk_poses=pose_arr,
            grippers=grip_arr,
        )
        try:
            response = self._request(
                {
                    "request_npz": request_npz,
                    "response_npy": response_npy,
                    "save_dir": save_dir,
                }
            )
            out = np.load(str(response.get("response_npy", response_npy))).astype(np.float32)
        finally:
            try:
                os.remove(request_npz)
                os.remove(response_npy)
            except OSError:
                pass
        return torch.from_numpy(out).float()

    def close(self) -> None:
        if self.execution_mode == "direct":
            runner = self.runner
            self.runner = None
            if runner is not None:
                try:
                    runner.close()
                except Exception:
                    pass
            return

        proc = self.proc
        self.proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                conn = Client(str(self.socket_path), family="AF_UNIX", authkey=b"smolvla_cosmos")
                try:
                    conn.send({"op": "shutdown"})
                    conn.recv()
                finally:
                    conn.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def cosmos_inference(
    cosmos_client: CosmosRolloutClient,
    curr_image: torch.Tensor,
    fk_poses: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    grippers: list[tuple[float, float]],
    raw_data: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    save_dir: str | None = None,
) -> torch.Tensor:
    _ = raw_data, device
    return cosmos_client.infer(curr_image, fk_poses, grippers, save_dir=save_dir)


def build_cosmos_client_from_cfg(cfg: dict[str, Any], device: str | torch.device = "cuda") -> CosmosRolloutClient:
    return CosmosRolloutClient(
        cosmos_root=str(cfg.get("cosmos_root", "/data/zhenyangfan/cosmos-predict2.5")),
        python_bin=str(cfg.get("cosmos_python_bin", "/data/zhenyangfan/cosmos-predict2.5/.venv/bin/python")),
        checkpoint_path=str(cfg["cosmos_checkpoint_path"]),
        experiment=str(cfg.get("cosmos_experiment", "robotwin_dualarm_actioncond_2b_256_320")),
        config_file=str(
            cfg.get(
                "cosmos_config_file",
                "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
            )
        ),
        device=str(device),
        cuda_visible_devices=str(cfg.get("cosmos_cuda_visible_devices", "") or ""),
        context_parallel_size=int(cfg.get("cosmos_context_parallel_size", 1)),
        chunk_size=int(cfg.get("cosmos_chunk_size", 12)),
        guidance=int(cfg.get("cosmos_guidance", 7)),
        resolution=str(cfg.get("cosmos_resolution", "256,320")),
        fps_downsample_ratio=int(cfg.get("cosmos_fps_downsample_ratio", 1)),
        gripper_scale=float(cfg.get("cosmos_gripper_scale", 1.0)),
        invert_gripper=bool(cfg.get("cosmos_invert_gripper", True)),
        num_steps=int(cfg.get("cosmos_num_steps", 35)),
        save_fps=int(cfg.get("cosmos_save_fps", 30)),
        num_latent_conditional_frames=int(cfg.get("cosmos_num_latent_conditional_frames", 1)),
        action_scaler=float(cfg.get("cosmos_action_scaler", 20.0)),
        action_stats_path=str(cfg.get("cosmos_action_stats_path", "")),
        action_normalization_clip=(
            None
            if cfg.get("cosmos_action_normalization_clip", None) in {None, "", "none", "None"}
            else float(cfg.get("cosmos_action_normalization_clip"))
        ),
        use_quat=bool(cfg.get("cosmos_use_quat", False)),
        prompt=str(cfg.get("cosmos_prompt", "")),
        negative_prompt=str(cfg.get("cosmos_negative_prompt", "")),
        seed=int(cfg.get("cosmos_seed", cfg.get("seed", 0))),
        work_dir=str(cfg.get("cosmos_work_dir", "")) or None,
        execution_mode=str(cfg.get("cosmos_execution_mode", "worker")),
        startup_timeout_s=float(cfg.get("cosmos_startup_timeout_s", 600.0)),
        request_timeout_s=float(cfg.get("cosmos_request_timeout_s", 900.0)),
    )


def write_cosmos_client_summary(client: CosmosRolloutClient, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "backend": "cosmos",
        "cosmos_root": client.cosmos_root,
        "python_bin": client.python_bin,
        "checkpoint_path": client.checkpoint_path,
        "experiment": client.experiment,
        "config_file": client.config_file,
        "resolution": client.resolution,
        "chunk_size": client.chunk_size,
        "fps_downsample_ratio": client.fps_downsample_ratio,
        "invert_gripper": client.invert_gripper,
        "action_scaler": client.action_scaler,
        "action_stats_path": client.action_stats_path,
        "action_normalization_clip": client.action_normalization_clip,
        "execution_mode": client.execution_mode,
        "worker_log_path": client.log_path,
        "worker_socket_path": client.socket_path,
    }
    with open(os.path.join(output_dir, "cosmos_client_meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
