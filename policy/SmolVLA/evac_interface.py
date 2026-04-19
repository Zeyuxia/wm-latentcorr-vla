from __future__ import annotations

import os
import sys
from math import ceil
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


class EvacLatentTeacher:
    """
    Minimal EVAC latent interface for SmolVLA latent training.

    This wrapper exposes final VAE latent encoding from real frames and the
    single-chunk rollout latent used by stage-2.
    """

    def __init__(self, evac_ckpt: str, evac_config: str, device: str | torch.device = "cuda:0"):
        self.device = torch.device(device)
        self.model, self.cfg = self._load_model(evac_ckpt, evac_config)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_model(self, evac_ckpt: str, evac_config: str) -> tuple[Any, Any]:
        if not os.path.exists(evac_ckpt):
            raise FileNotFoundError(f"EVAC checkpoint not found: {evac_ckpt}")
        if not os.path.exists(evac_config):
            raise FileNotFoundError(f"EVAC config not found: {evac_config}")

        config = OmegaConf.load(evac_config)
        config.model.pretrained_checkpoint = evac_ckpt

        evac_pkg_root = os.path.realpath(os.path.join(os.path.dirname(evac_config), "..", ".."))
        evac_module_root = os.path.join(evac_pkg_root, "evac")
        for path in (evac_module_root, evac_pkg_root):
            if path not in sys.path:
                sys.path.insert(0, path)

        from evac.utils.general_utils import instantiate_from_config, load_checkpoints  # noqa: WPS433

        model = instantiate_from_config(config.model)
        model = load_checkpoints(model, config.model, ignore_mismatched_sizes=False)
        model = model.to(self.device)
        return model, config

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must be (B,3,H,W), got {image.shape}")
        image = image.to(device=self.device, dtype=torch.float32)
        sample_h, sample_w = tuple(self.cfg.data.params.train.params.sample_size)
        image = F.interpolate(image, size=(sample_h, sample_w), mode="bilinear", align_corners=False)
        image = (image * 2.0) - 1.0
        latent = self.model.encode_first_stage(image, mode=True)
        if latent.ndim != 4:
            raise RuntimeError(f"Expected 4D latent, got shape {tuple(latent.shape)}")
        return latent.float()

    @torch.no_grad()
    def _prepare_rollout_inputs(
        self,
        curr_image: torch.Tensor,
        curr_qpos_raw: torch.Tensor,
        action_prefix_raw: torch.Tensor,
        raw_data: dict[str, Any],
        fk,
        ddim_steps: int = 27,
        dataset_name: str = "agibotworld",
        inference_dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor | int]:
        from einops import rearrange
        import torchvision.transforms as tvt

        from evac.lvdm.data.get_actions import get_actions
        from evac.lvdm.data.statistics import StatisticInfo

        if curr_image.ndim != 3:
            raise ValueError(f"curr_image must be (3,H,W), got {tuple(curr_image.shape)}")
        if curr_qpos_raw.ndim == 2:
            if curr_qpos_raw.shape[0] != 1:
                raise ValueError(f"curr_qpos_raw batch dim must be 1, got {tuple(curr_qpos_raw.shape)}")
            curr_qpos_raw = curr_qpos_raw[0]
        if action_prefix_raw.ndim == 3:
            if action_prefix_raw.shape[0] != 1:
                raise ValueError(f"action_prefix_raw batch dim must be 1, got {tuple(action_prefix_raw.shape)}")
            action_prefix_raw = action_prefix_raw[0]
        if action_prefix_raw.ndim != 2:
            raise ValueError(f"action_prefix_raw must be (K,14), got {tuple(action_prefix_raw.shape)}")

        chunk = int(self.cfg.chunk)
        n_previous = int(self.cfg.n_previous)
        n_valid = int(action_prefix_raw.shape[0])
        num_chunk = int(ceil(float(max(1, n_valid)) / float(chunk)))
        if num_chunk != 1:
            raise NotImplementedError(
                f"Minimal stage-2 rollout currently supports a single EVAC chunk only, got num_chunk={num_chunk}"
            )

        curr_q = curr_qpos_raw.detach().cpu().float().numpy().astype(np.float32)
        act_raw = action_prefix_raw.detach().cpu().float().numpy().astype(np.float32)

        fk_poses = []
        grippers = []
        left_q, right_q = curr_q[:6], curr_q[7:13]
        left_grip, right_grip = float(curr_q[6]), float(curr_q[13])
        fk_now = fk.forward(left_q, right_q)
        fk_poses.append((fk_now["left"][0], fk_now["left"][1], fk_now["right"][0], fk_now["right"][1]))
        grippers.append((left_grip, right_grip))
        for action in act_raw:
            fk_state = fk.forward(action[:6], action[7:13])
            fk_poses.append((fk_state["left"][0], fk_state["left"][1], fk_state["right"][0], fk_state["right"][1]))
            grippers.append((float(action[6]), float(action[13])))

        n_states = len(fk_poses)
        all_ends_p = np.zeros((n_states, 2, 3), dtype=np.float32)
        all_ends_o = np.zeros((n_states, 2, 4), dtype=np.float32)
        gripper_arr = np.zeros((n_states, 2), dtype=np.float32)
        for idx, ((left_p, left_qt, right_p, right_qt), (left_g, right_g)) in enumerate(zip(fk_poses, grippers)):
            all_ends_p[idx, 0], all_ends_p[idx, 1] = left_p, right_p
            left_q_xyzw = np.array([left_qt[1], left_qt[2], left_qt[3], left_qt[0]], dtype=np.float32)
            right_q_xyzw = np.array([right_qt[1], right_qt[2], right_qt[3], right_qt[0]], dtype=np.float32)
            if left_q_xyzw[3] < 0:
                left_q_xyzw = -left_q_xyzw
            if right_q_xyzw[3] < 0:
                right_q_xyzw = -right_q_xyzw
            all_ends_o[idx, 0] = left_q_xyzw
            all_ends_o[idx, 1] = right_q_xyzw
            gripper_arr[idx] = [float(left_g) * 120.0, float(right_g) * 120.0]

        slices = [0] * (n_previous - 1) + list(range(n_states))
        action_abs, delta_action = get_actions(
            gripper=gripper_arr,
            all_ends_p=all_ends_p,
            all_ends_o=all_ends_o,
            slices=slices,
            delta_act_sidx=n_previous,
        )

        action_abs = torch.from_numpy(action_abs).float()
        delta_action = torch.from_numpy(delta_action).float()
        stat_mean = torch.tensor(StatisticInfo[dataset_name]["mean"]).unsqueeze(0)
        stat_std = torch.tensor(StatisticInfo[dataset_name]["std"]).unsqueeze(0)
        delta_action[:, :6] = (delta_action[:, :6] - stat_mean[:, :6]) / stat_std[:, :6]
        delta_action[:, 7:13] = (delta_action[:, 7:13] - stat_mean[:, 6:]) / stat_std[:, 6:]

        h_native, w_native = raw_data["native_resolution"]
        img_rgb = curr_image[[2, 1, 0]]
        img_rgb = tvt.Resize((h_native, w_native))(img_rgb)
        memories = img_rgb.unsqueeze(1).repeat(1, n_previous, 1, 1)

        ext_cv = raw_data["extrinsic_cv"]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = ext_cv
        c2w = np.linalg.inv(w2c).astype(np.float32)

        n_action = action_abs.shape[0]
        c2w_t = torch.from_numpy(c2w).float().unsqueeze(0).unsqueeze(0).repeat(1, n_action, 1, 1)
        w2c_t = torch.from_numpy(w2c).float().unsqueeze(0).unsqueeze(0).repeat(1, n_action, 1, 1)
        intrinsic = torch.from_numpy(raw_data["intrinsic_cv"]).float().clone()

        sample_size = tuple(self.cfg.data.params.train.params.sample_size)
        trans_resize = tvt.Resize(sample_size)
        trans_norm = tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

        video_list = memories.unsqueeze(1)
        video_list = rearrange(video_list, "c v t h w -> (v t) c h w")
        video_list = trans_resize(video_list)
        video_list = trans_norm(video_list)
        video_list = rearrange(video_list, "(v t) c h w -> c v t h w", v=1).unsqueeze(0)

        intrinsic_scaled = intrinsic.unsqueeze(0)
        h_scale = float(sample_size[0]) / float(h_native)
        w_scale = float(sample_size[1]) / float(w_native)
        intrinsic_scaled[:, 0, 0] *= w_scale
        intrinsic_scaled[:, 0, 2] *= w_scale
        intrinsic_scaled[:, 1, 1] *= h_scale
        intrinsic_scaled[:, 1, 2] *= h_scale

        traj_list = self.model.get_traj(sample_size, action_abs, w2c_t, c2w_t, intrinsic_scaled)
        traj_list = rearrange(traj_list, "c v t h w -> (v t) c h w")
        traj_list = trans_norm(traj_list)
        traj_list = rearrange(traj_list, "(v t) c h w -> c v t h w", v=1).unsqueeze(0)

        video = torch.cat(
            (
                video_list[:, :, :, :n_previous],
                video_list[:, :, :, n_previous - 1 : n_previous].repeat(1, 1, 1, chunk, 1, 1),
            ),
            dim=3,
        )
        traj = traj_list[:, :, :, : chunk + n_previous]
        i_delta_action = delta_action.unsqueeze(0)[:, :chunk]
        i_c2w_list = c2w_t.unsqueeze(0)[:, :, : chunk + n_previous]
        if traj.shape[3] < chunk + n_previous:
            pad_t = chunk + n_previous - traj.shape[3]
            traj = torch.cat((traj, traj[:, :, :, -1:].repeat(1, 1, 1, pad_t, 1, 1)), dim=3)
            i_delta_action = torch.cat(
                (
                    i_delta_action,
                    torch.zeros(
                        (i_delta_action.shape[0], chunk - i_delta_action.shape[1], i_delta_action.shape[2]),
                        dtype=i_delta_action.dtype,
                    ),
                ),
                dim=1,
            )
            i_c2w_list = torch.cat((i_c2w_list, i_c2w_list[:, :, -1:].repeat(1, 1, pad_t, 1, 1)), dim=2)

        video = torch.clamp(video, min=-1.0, max=1.0)
        traj = torch.clamp(traj, min=-1.0, max=1.0)
        last_idx = int(n_previous + n_valid - 1)
        return {
            "video": video,
            "traj": traj,
            "delta_action": i_delta_action,
            "intrinsic": intrinsic_scaled.unsqueeze(0),
            "extrinsic": i_c2w_list,
            "last_idx": last_idx,
        }

    @torch.no_grad()
    def rollout_latent_from_actions_batch(
        self,
        curr_image: torch.Tensor,
        curr_qpos_raw: torch.Tensor,
        action_prefix_raw: torch.Tensor,
        raw_data: list[dict[str, Any]] | dict[str, Any],
        fk,
        ddim_steps: int = 27,
        dataset_name: str = "agibotworld",
        inference_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        from evac.lvdm.data.domain_table import DomainTable

        if curr_image.ndim == 3:
            curr_image = curr_image.unsqueeze(0)
        if curr_qpos_raw.ndim == 1:
            curr_qpos_raw = curr_qpos_raw.unsqueeze(0)
        if action_prefix_raw.ndim == 2:
            action_prefix_raw = action_prefix_raw.unsqueeze(0)
        if curr_image.ndim != 4:
            raise ValueError(f"curr_image must be (B,3,H,W), got {tuple(curr_image.shape)}")
        if curr_qpos_raw.ndim != 2:
            raise ValueError(f"curr_qpos_raw must be (B,14), got {tuple(curr_qpos_raw.shape)}")
        if action_prefix_raw.ndim != 3:
            raise ValueError(f"action_prefix_raw must be (B,K,14), got {tuple(action_prefix_raw.shape)}")

        batch_size = int(curr_image.shape[0])
        if curr_qpos_raw.shape[0] != batch_size or action_prefix_raw.shape[0] != batch_size:
            raise ValueError(
                "curr_image, curr_qpos_raw, and action_prefix_raw batch sizes must match: "
                f"{tuple(curr_image.shape)}, {tuple(curr_qpos_raw.shape)}, {tuple(action_prefix_raw.shape)}"
            )
        if isinstance(raw_data, dict):
            raw_data_list = [raw_data] * batch_size
        else:
            raw_data_list = list(raw_data)
        if len(raw_data_list) != batch_size:
            raise ValueError(f"raw_data length must match batch size {batch_size}, got {len(raw_data_list)}")

        chunk = int(self.cfg.chunk)
        prepared = [
            self._prepare_rollout_inputs(
                curr_image=curr_image[i],
                curr_qpos_raw=curr_qpos_raw[i],
                action_prefix_raw=action_prefix_raw[i],
                raw_data=raw_data_list[i],
                fk=fk,
                ddim_steps=ddim_steps,
                dataset_name=dataset_name,
                inference_dtype=inference_dtype,
            )
            for i in range(batch_size)
        ]

        video = torch.cat([item["video"] for item in prepared], dim=0).to(device=self.device, dtype=inference_dtype)
        traj = torch.cat([item["traj"] for item in prepared], dim=0).to(device=self.device, dtype=inference_dtype)
        delta_action = torch.cat([item["delta_action"] for item in prepared], dim=0).to(
            device=self.device, dtype=inference_dtype
        )
        intrinsic = torch.cat([item["intrinsic"] for item in prepared], dim=0).to(device=self.device)
        extrinsic = torch.cat([item["extrinsic"] for item in prepared], dim=0).to(device=self.device)
        last_indices = [int(item["last_idx"]) for item in prepared]

        fps = 30 * torch.ones((batch_size,), device=self.device)
        domain_id = torch.full(
            (batch_size,),
            int(DomainTable[dataset_name]),
            dtype=torch.long,
            device=self.device,
        )
        batch = {
            "video": video,
            "traj": traj,
            "delta_action": delta_action,
            "domain_id": domain_id,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "caption": [""] * batch_size,
            "cond_id": torch.full(
                (batch_size,),
                int(-self.cfg.n_previous - self.cfg.chunk),
                dtype=torch.int64,
                device=self.device,
            ),
            "fps": fps,
        }

        self.model._dc_ddim_sampler = None
        self.model.ddim_num_chunk = 1
        self.model.rand_cond_frame = False

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z, cond, _, fs, did, _ = self.model.get_batch_input(
                batch,
                random_uncond=False,
                return_first_stage_outputs=False,
                return_original_cond=True,
                return_fs=True,
                return_did=True,
                return_traj=False,
                return_img_emb=True,
            )
            kwargs = {"fs": fs.long(), "domain_id": did.long()}
            for idx in range(len(cond["c_concat"])):
                cond["c_concat"][idx] = cond["c_concat"][idx].to(dtype=inference_dtype)
            for idx in range(len(cond["c_crossattn"])):
                cond["c_crossattn"][idx] = cond["c_crossattn"][idx].to(dtype=inference_dtype)
            samples, _ = self.model.sample_log(
                cond=cond,
                batch_size=z.shape[0],
                ddim=True,
                ddim_steps=ddim_steps,
                causal=True,
                eta=1.0,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
                x0=z.to(inference_dtype),
                chunk=chunk,
                cat_mask=self.model.use_cat_mask,
                sparse=self.model.sparse_memory,
                traj=False,
                ddim_dtype=torch.float16,
                timestep_spacing="uniform_trailing",
                dtype=inference_dtype,
                return_intermediates=False,
                **kwargs,
            )

        latent = torch.cat(
            [samples[i : i + 1, :, last_indices[i]].float() for i in range(batch_size)],
            dim=0,
        )
        if latent.ndim != 4:
            raise RuntimeError(f"Expected 4D latent rollout result, got {tuple(latent.shape)}")
        return latent

    @torch.no_grad()
    def rollout_latent_from_actions(
        self,
        curr_image: torch.Tensor,
        curr_qpos_raw: torch.Tensor,
        action_prefix_raw: torch.Tensor,
        raw_data: dict[str, Any],
        fk,
        ddim_steps: int = 27,
        dataset_name: str = "agibotworld",
        inference_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return self.rollout_latent_from_actions_batch(
            curr_image=curr_image,
            curr_qpos_raw=curr_qpos_raw,
            action_prefix_raw=action_prefix_raw,
            raw_data=raw_data,
            fk=fk,
            ddim_steps=ddim_steps,
            dataset_name=dataset_name,
            inference_dtype=inference_dtype,
        )
