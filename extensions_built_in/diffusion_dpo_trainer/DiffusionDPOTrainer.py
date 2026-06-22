from __future__ import annotations

import contextlib
import sqlite3
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.config_modules import GenerateImageConfig
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


DIFFUSION_DPO_TRAINER_TYPE = "diffusion_dpo_trainer"
PAIR_SIZE = 2


@dataclass
class DiffusionDPOConfig:
    objective: Literal["classic", "linear"] = "classic"
    beta: float = 5000.0

    def __init__(self, **kwargs):
        objective = kwargs.get("objective", self.objective)
        if objective not in {"classic", "linear"}:
            raise ValueError("Diffusion-DPO objective must be 'classic' or 'linear'.")
        self.objective = objective
        self.beta = float(kwargs.get("beta", self.beta))


class DiffusionDPOTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        train_config = config.setdefault("train", {})
        train_config["disable_sampling"] = True
        sample_config = config.setdefault("sample", {})
        sample_config["sample_every"] = 0
        sample_config["samples"] = []
        super().__init__(process_id, job, config, **kwargs)
        if not self.is_ui_trainer:
            raise ValueError("diffusion_dpo_trainer requires the UI SQLite runtime.")

        self.dpo_config = DiffusionDPOConfig(**self.config.get("dpo", {}))
        self._task_counter = 0
        self._dpo_root = Path(self.save_root) / "diffusion_dpo"
        self._candidate_root = self._dpo_root / "candidates"
        self._candidate_root.mkdir(parents=True, exist_ok=True)
        self._ensure_voting_schema()

    def _db_execute(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        def _op():
            with self._db_connect() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()

        return self._retry_db_operation(_op)

    def _db_execute_write(self, query: str, params: tuple = ()) -> None:
        def _op():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

        self._retry_db_operation(_op)

    def _db_execute_many(self, query: str, rows: list[tuple]) -> None:
        def _op():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, rows)

        self._retry_db_operation(_op)

    def _ensure_voting_schema(self) -> None:
        for table in ("FlowGRPOVoteTask", "FlowGRPOCandidate", "FlowGRPOVote"):
            columns = self._db_execute(f"PRAGMA table_info({table})")
            column_names = {row["name"] for row in columns}
            if "trainer_type" not in column_names:
                self._db_execute_write(
                    f"ALTER TABLE {table} ADD COLUMN trainer_type TEXT NOT NULL DEFAULT 'flow_grpo_trainer'"
                )
        candidate_columns = self._db_execute("PRAGMA table_info(FlowGRPOCandidate)")
        candidate_column_names = {row["name"] for row in candidate_columns}
        if "state_path" not in candidate_column_names:
            self._db_execute_write("ALTER TABLE FlowGRPOCandidate ADD COLUMN state_path TEXT NOT NULL DEFAULT ''")

    def _recover_stale_generating_tasks(self) -> None:
        rows = self._db_execute(
            """
            SELECT id FROM FlowGRPOVoteTask
            WHERE job_id = ? AND trainer_type = ? AND status = 'generating'
            ORDER BY created_at ASC
            """,
            (self.job_id, DIFFUSION_DPO_TRAINER_TYPE),
        )
        for row in rows:
            self._db_execute_write(
                "UPDATE FlowGRPOVoteTask SET status = 'failed', error = ? WHERE id = ? AND job_id = ?",
                ("Generation was stopped before completion.", row["id"], self.job_id),
            )
            self._db_execute_write(
                "UPDATE FlowGRPOCandidate SET status = 'failed' WHERE vote_task_id = ? AND job_id = ?",
                (row["id"], self.job_id),
            )

    def _count_open_tasks(self) -> int:
        rows = self._db_execute(
            """
            SELECT COUNT(*) AS count FROM FlowGRPOVoteTask
            WHERE job_id = ? AND trainer_type = ? AND status = 'open'
            """,
            (self.job_id, DIFFUSION_DPO_TRAINER_TYPE),
        )
        return int(rows[0]["count"]) if rows else 0

    def _count_active_vote_tasks(self) -> int:
        rows = self._db_execute(
            """
            SELECT COUNT(*) AS count FROM FlowGRPOVoteTask
            WHERE job_id = ? AND trainer_type = ? AND status IN ('generating', 'open')
            """,
            (self.job_id, DIFFUSION_DPO_TRAINER_TYPE),
        )
        return int(rows[0]["count"]) if rows else 0

    def _task_dir(self, task_id: str) -> Path:
        task_dir = self._candidate_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _candidate_seed(self, task_row: sqlite3.Row, order_index: int) -> int:
        task_seed = task_row["seed"]
        if task_seed is not None:
            return int(task_seed) + order_index
        return int(self.sample_config.seed) + order_index + self.step_num + (self._task_counter * 1000)

    def _build_image_config(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        image_path: Path,
    ) -> GenerateImageConfig:
        return GenerateImageConfig(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            network_multiplier=getattr(self.sample_config, "network_multiplier", 1.0),
            output_path=str(image_path),
            output_ext=image_path.suffix.lstrip("."),
            logger=self.logger,
            num_frames=getattr(self.sample_config, "num_frames", 1),
            fps=getattr(self.sample_config, "fps", 1),
            guidance_rescale=getattr(self.sample_config, "guidance_rescale", 0.0),
            adapter_conditioning_scale=getattr(self.sample_config, "adapter_conditioning_scale", 1.0),
            refiner_start_at=getattr(self.sample_config, "refiner_start_at", 0.5),
            extra_values=getattr(self.sample_config, "extra_values", []),
            ctrl_img=getattr(self.sample_config, "ctrl_img", None),
            ctrl_img_1=getattr(self.sample_config, "ctrl_img_1", None),
            ctrl_img_2=getattr(self.sample_config, "ctrl_img_2", None),
            ctrl_img_3=getattr(self.sample_config, "ctrl_img_3", None),
            ctrl_idx=getattr(self.sample_config, "ctrl_idx", 0),
            do_cfg_norm=getattr(self.sample_config, "do_cfg_norm", False),
        )

    def _next_requested_task(self) -> Optional[sqlite3.Row]:
        rows = self._db_execute(
            """
            SELECT * FROM FlowGRPOVoteTask
            WHERE job_id = ? AND trainer_type = ? AND status = 'requested'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (self.job_id, DIFFUSION_DPO_TRAINER_TYPE),
        )
        return rows[0] if rows else None

    def _generate_requested_task(self, task_row: sqlite3.Row) -> None:
        task_id = task_row["id"]
        prompt = task_row["prompt"] or ""
        negative_prompt = task_row["negative_prompt"] or ""
        width = int(task_row["width"] or self.sample_config.width)
        height = int(task_row["height"] or self.sample_config.height)
        guidance_scale = float(task_row["guidance_scale"] or self.sample_config.guidance_scale)
        num_inference_steps = int(task_row["num_inference_steps"] or self.sample_config.sample_steps)
        sampler = task_row["sampler"] or self.sample_config.sampler
        task_dir = self._task_dir(task_id)

        self._db_execute_write(
            "UPDATE FlowGRPOVoteTask SET status = 'generating' WHERE id = ? AND job_id = ?",
            (task_id, self.job_id),
        )

        try:
            self.update_status("running", "Generating Diffusion-DPO pair")
            image_configs: list[GenerateImageConfig] = []
            candidate_rows: list[tuple] = []
            for order_index in range(PAIR_SIZE):
                self.maybe_stop()
                candidate_id = str(uuid.uuid4())
                seed = self._candidate_seed(task_row, order_index)
                image_path = task_dir / f"{order_index:02d}_{candidate_id}.png"
                image_config = self._build_image_config(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    image_path=image_path,
                )
                image_configs.append(image_config)
                candidate_rows.append(
                    (
                        candidate_id,
                        self.job_id,
                        task_id,
                        DIFFUSION_DPO_TRAINER_TYPE,
                        order_index,
                        image_config.prompt,
                        image_config.negative_prompt,
                        seed,
                        guidance_scale,
                        num_inference_steps,
                        sampler,
                        task_row["scheduler"] or self.train_config.noise_scheduler or "",
                        str(image_path),
                        "",
                        "open",
                    )
                )

            self.sd.generate_images(image_configs, sampler=sampler)
            self._db_execute_many(
                """
                INSERT INTO FlowGRPOCandidate (
                    id, job_id, vote_task_id, trainer_type, order_index, prompt, negative_prompt, seed,
                    guidance_scale, num_inference_steps, sampler, scheduler, image_path, state_path, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                candidate_rows,
            )
            self._db_execute_write(
                "UPDATE FlowGRPOVoteTask SET status = 'open', error = NULL WHERE id = ? AND job_id = ?",
                (task_id, self.job_id),
            )
            self._task_counter += 1
        except Exception as exc:
            self._db_execute_write(
                "UPDATE FlowGRPOVoteTask SET status = 'failed', error = ? WHERE id = ? AND job_id = ?",
                (str(exc), task_id, self.job_id),
            )
            self._db_execute_write(
                "UPDATE FlowGRPOCandidate SET status = 'failed' WHERE vote_task_id = ?",
                (task_id,),
            )
            raise

    def _promote_requested_vote_tasks(self) -> None:
        active_count = self._count_active_vote_tasks()
        while active_count < 1:
            task_row = self._next_requested_task()
            if task_row is None:
                return
            self._generate_requested_task(task_row)
            active_count += 1

    def _next_voted_task(self) -> Optional[str]:
        rows = self._db_execute(
            """
            SELECT task.id
            FROM FlowGRPOVoteTask task
            WHERE task.job_id = ?
              AND task.trainer_type = ?
              AND task.status = 'voted'
              AND (
                SELECT COUNT(*)
                FROM FlowGRPOCandidate candidate
                WHERE candidate.vote_task_id = task.id
              ) = 2
            ORDER BY task.created_at ASC
            LIMIT 1
            """,
            (self.job_id, DIFFUSION_DPO_TRAINER_TYPE),
        )
        return rows[0]["id"] if rows else None

    def _load_task_rows(self, task_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row], list[sqlite3.Row]]:
        task_rows = self._db_execute(
            "SELECT * FROM FlowGRPOVoteTask WHERE id = ? AND job_id = ? AND trainer_type = ? LIMIT 1",
            (task_id, self.job_id, DIFFUSION_DPO_TRAINER_TYPE),
        )
        if not task_rows:
            raise ValueError(f"Diffusion-DPO vote task '{task_id}' was not found.")
        candidate_rows = self._db_execute(
            "SELECT * FROM FlowGRPOCandidate WHERE vote_task_id = ? ORDER BY order_index ASC",
            (task_id,),
        )
        vote_rows = self._db_execute(
            "SELECT * FROM FlowGRPOVote WHERE vote_task_id = ? AND processed = 0 ORDER BY created_at ASC",
            (task_id,),
        )
        return task_rows[0], candidate_rows, vote_rows

    def _mark_task_processed(self, task_id: str) -> None:
        self._db_execute_write("UPDATE FlowGRPOVoteTask SET status = 'processed' WHERE id = ?", (task_id,))
        self._db_execute_write("UPDATE FlowGRPOCandidate SET status = 'processed' WHERE vote_task_id = ?", (task_id,))
        self._db_execute_write("UPDATE FlowGRPOVote SET processed = 1 WHERE vote_task_id = ?", (task_id,))

    @contextlib.contextmanager
    def _network_disabled(self):
        if self.network is None:
            yield
            return
        was_active = bool(getattr(self.network, "is_active", True))
        self.network.is_active = False
        try:
            yield
        finally:
            self.network.is_active = was_active

    def _load_image_latents(self, image_path: str) -> torch.Tensor:
        dtype = get_torch_dtype(self.train_config.dtype)
        image = Image.open(image_path).convert("RGB")
        tensor = transforms.ToTensor()(image).to(self.device_torch, dtype=dtype)
        tensor = (tensor * 2.0) - 1.0
        with torch.no_grad():
            latents = self.sd.encode_images([tensor]).to(self.device_torch, dtype=dtype)
        return latents * self.train_config.latent_multiplier

    def _setup_train_timesteps(self, latents: torch.Tensor) -> None:
        num_train_timesteps = int(self.train_config.num_train_timesteps)
        if self.train_config.noise_scheduler == "flowmatch":
            timestep_type = self.train_config.timestep_type
            if any(
                [
                    self.train_config.linear_timesteps,
                    self.train_config.linear_timesteps2,
                    timestep_type == "linear",
                    timestep_type in ["one_step", "two_step", "four_step", "eight_step"],
                ]
            ):
                timestep_type = "linear"
            patch_size = 1
            if getattr(self.sd, "is_flux", False) or "flex" in getattr(self.sd, "arch", ""):
                patch_size = 2
            elif hasattr(getattr(self.sd, "unet", None), "config") and hasattr(self.sd.unet.config, "patch_size"):
                patch_size = self.sd.unet.config.patch_size
            self.sd.noise_scheduler.set_train_timesteps(
                num_train_timesteps,
                device=self.device_torch,
                timestep_type=timestep_type,
                latents=latents,
                patch_size=patch_size,
            )
        elif self.train_config.noise_scheduler == "lcm":
            self.sd.noise_scheduler.set_timesteps(
                num_train_timesteps,
                device=self.device_torch,
                original_inference_steps=num_train_timesteps,
            )
        else:
            self.sd.noise_scheduler.set_timesteps(num_train_timesteps, device=self.device_torch)

    def _sample_shared_timestep(self, latents: torch.Tensor) -> torch.Tensor:
        self._setup_train_timesteps(latents)
        max_index = max(1, len(self.sd.noise_scheduler.timesteps) - 1)
        min_step = max(0, int(self.train_config.min_denoising_steps))
        max_step = min(max_index, int(self.train_config.max_denoising_steps), max_index)
        if self.train_config.timestep_type == "one_step" or min_step >= max_step:
            timestep_index = torch.tensor([min_step], device=self.device_torch, dtype=torch.long)
        else:
            timestep_index = torch.randint(min_step, max_step, (1,), device=self.device_torch, dtype=torch.long)
        return self.sd.noise_scheduler.timesteps[timestep_index.long()]

    def _encode_prompt_pair(self, prompt: str, negative_prompt: str) -> tuple[PromptEmbeds, Optional[PromptEmbeds]]:
        conditional_one = self.sd.encode_prompt(prompt, force_all=True)
        conditional = concat_prompt_embeds([conditional_one, conditional_one]).to(
            self.device_torch,
            dtype=get_torch_dtype(self.train_config.dtype),
        )
        unconditional = None
        if self.train_config.do_cfg or self.train_config.do_random_cfg:
            unconditional_one = self.sd.encode_prompt(negative_prompt, force_all=True)
            unconditional = concat_prompt_embeds([unconditional_one, unconditional_one]).to(
                self.device_torch,
                dtype=get_torch_dtype(self.train_config.dtype),
            )
        return conditional, unconditional

    def _dpo_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        fake_batch = SimpleNamespace(latents=latents, tensor=latents)
        if hasattr(self.sd, "get_loss_target"):
            return self.sd.get_loss_target(noise=noise, batch=fake_batch, timesteps=timesteps).detach()
        if self.sd.prediction_type == "v_prediction":
            return self.sd.noise_scheduler.get_velocity(latents, noise, timesteps).detach()
        if self.sd.is_flow_matching:
            return (noise - latents).detach()
        return noise.detach()

    def _pair_losses(
        self,
        *,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        target: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: Optional[PromptEmbeds],
        disable_network: bool,
    ) -> torch.Tensor:
        dtype = get_torch_dtype(self.train_config.dtype)
        fake_batch = SimpleNamespace(
            latents=latents,
            tensor=None,
            control_tensor=None,
            control_tensor_list=None,
            inpaint_tensor=None,
        )
        noisy_latents = self.sd.add_noise(latents, noise, timesteps)
        model_latents = self.sd.condition_noisy_latents(noisy_latents, fake_batch)
        with contextlib.ExitStack() as stack:
            if disable_network:
                stack.enter_context(torch.no_grad())
                stack.enter_context(self._network_disabled())
            pred = self.predict_noise(
                noisy_latents=model_latents.to(self.device_torch, dtype=dtype),
                timesteps=timesteps,
                conditional_embeds=conditional_embeds,
                unconditional_embeds=unconditional_embeds,
                batch=fake_batch,
                is_primary_pred=not disable_network,
            )
        losses = F.mse_loss(pred.float(), target.float(), reduction="none")
        return losses.mean(dim=tuple(range(1, losses.ndim)))

    def _process_voted_task(self, task_id: str) -> Optional[dict[str, float]]:
        task_row, candidate_rows, vote_rows = self._load_task_rows(task_id)
        if len(candidate_rows) != PAIR_SIZE:
            self._mark_task_processed(task_id)
            return None

        reward_by_candidate = {row["candidate_id"]: float(row["reward"]) for row in vote_rows}
        if len(reward_by_candidate) != PAIR_SIZE:
            return None

        left, right = candidate_rows
        reward_left = reward_by_candidate.get(left["id"], 0.0)
        reward_right = reward_by_candidate.get(right["id"], 0.0)
        if reward_left == reward_right:
            self._mark_task_processed(task_id)
            return None
        winner, loser = (left, right) if reward_left > reward_right else (right, left)

        if self.network is None:
            raise ValueError("Diffusion-DPO trainer requires a trainable LoRA target.")

        dtype = get_torch_dtype(self.train_config.dtype)
        self.network.train()
        self.network.is_active = True
        self.optimizer.zero_grad(set_to_none=True)

        latents_w = self._load_image_latents(winner["image_path"])
        latents_l = self._load_image_latents(loser["image_path"])
        latents = torch.cat([latents_w, latents_l], dim=0).to(self.device_torch, dtype=dtype)

        noise_one = self.sd.get_latent_noise_from_latents(
            latents_w.to(self.device_torch, dtype=dtype),
            noise_offset=self.train_config.noise_offset,
        ).to(self.device_torch, dtype=dtype)
        noise = torch.cat([noise_one, noise_one], dim=0)
        t_one = self._sample_shared_timestep(latents_w)
        timesteps = torch.cat([t_one, t_one], dim=0)
        target = self._dpo_target(latents, noise, timesteps)
        conditional_embeds, unconditional_embeds = self._encode_prompt_pair(
            task_row["prompt"] or "",
            task_row["negative_prompt"] or "",
        )

        policy_losses = self._pair_losses(
            latents=latents,
            noise=noise,
            timesteps=timesteps,
            target=target,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            disable_network=False,
        )
        policy_w, policy_l = policy_losses.chunk(2)

        ref_losses = self._pair_losses(
            latents=latents,
            noise=noise,
            timesteps=timesteps,
            target=target,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            disable_network=True,
        )
        ref_w, ref_l = ref_losses.chunk(2)

        model_diff = policy_w - policy_l
        ref_diff = ref_w - ref_l
        inside_term = -0.5 * self.dpo_config.beta * (model_diff - ref_diff)
        implicit_acc = (inside_term > 0).float().mean()
        if self.dpo_config.objective == "classic":
            loss = -F.logsigmoid(inside_term).mean()
        else:
            loss = -inside_term.mean()

        self.accelerator.backward(loss)
        if self.params and self.train_config.optimizer != "adafactor":
            if isinstance(self.params[0], dict):
                for param_group in self.params:
                    self.accelerator.clip_grad_norm_(param_group["params"], self.train_config.max_grad_norm)
            else:
                self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
        self.optimizer.step()
        if self.adapter is not None and hasattr(self.adapter, "post_weight_update"):
            self.adapter.post_weight_update()
        if self.ema is not None:
            self.ema.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._mark_task_processed(task_id)

        return {
            "loss": float(loss.detach().cpu().item()),
            "policy_w": float(policy_w.detach().cpu().item()),
            "policy_l": float(policy_l.detach().cpu().item()),
            "ref_w": float(ref_w.detach().cpu().item()),
            "ref_l": float(ref_l.detach().cpu().item()),
            "model_mse": float((0.5 * (policy_w.mean() + policy_l.mean())).detach().cpu().item()),
            "ref_mse": float(ref_losses.mean().detach().cpu().item()),
            "implicit_acc": float(implicit_acc.detach().cpu().item()),
            "dpo_margin": float(inside_term.detach().mean().cpu().item()),
        }

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self._recover_stale_generating_tasks()
        self._promote_requested_vote_tasks()

    def hook_train_loop(self, batch):
        while True:
            self.maybe_stop()
            self._promote_requested_vote_tasks()
            task_id = self._next_voted_task()
            if task_id is None:
                self.update_status("running", "Waiting for Diffusion-DPO pair votes")
                time.sleep(2.0)
                continue

            self.update_status("running", "Applying Diffusion-DPO pair vote")
            metrics = self._process_voted_task(task_id)
            if metrics is None:
                continue
            self.update_status("running", "Waiting for Diffusion-DPO pair votes")
            return metrics

    def get_training_info(self):
        info = super().get_training_info()
        info["diffusion_dpo_open_tasks"] = self._count_open_tasks()
        return info
