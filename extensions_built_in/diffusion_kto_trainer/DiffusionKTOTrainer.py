from __future__ import annotations

import contextlib
import json
import math
import os
import random
import sqlite3
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import transforms

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.config_modules import GenerateImageConfig, NetworkConfig
from toolkit.dataloader_mixins import clean_caption, get_comfyui_caption_from_png_metadata
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.print import print_acc
from toolkit.train_tools import get_torch_dtype


DIFFUSION_KTO_TRAINER_TYPE = "diffusion_kto_trainer"


@dataclass
class DiffusionKTOConfig:
    beta: float = 1000.0
    lambda_d: float = 1.0
    lambda_u: float = 1.0
    positive_ratio: float = 0.5
    halo: str = "sigmoid"
    bce_offset: str = "none"
    group_size: int = 4
    dataset_enabled: bool = False

    def __init__(self, **kwargs):
        self.beta = float(kwargs.get("beta", kwargs.get("beta_dpo", self.beta)))
        self.lambda_d = float(kwargs.get("lambda_d", kwargs.get("lambda_d_kto", self.lambda_d)))
        self.lambda_u = float(kwargs.get("lambda_u", kwargs.get("lambda_u_kto", self.lambda_u)))
        self.positive_ratio = min(1.0, max(0.0, float(kwargs.get("positive_ratio", self.positive_ratio))))
        self.halo = str(kwargs.get("halo", self.halo)).lower()
        if self.halo != "sigmoid":
            raise ValueError("Diffusion-KTO currently supports halo='sigmoid'.")
        self.bce_offset = str(kwargs.get("bce_offset", self.bce_offset)).lower()
        if self.bce_offset not in {"none", "sigmoid", "original"}:
            raise ValueError("Diffusion-KTO bce_offset must be 'none', 'sigmoid', or 'original'.")
        self.group_size = max(1, int(kwargs.get("group_size", self.group_size)))
        self.dataset_enabled = bool(kwargs.get("dataset_enabled", kwargs.get("use_dataset", self.dataset_enabled)))


class DiffusionKTOTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        config.setdefault("kto", {})
        kto_config = DiffusionKTOConfig(**config.get("kto", {}))
        model_config = config.setdefault("model", {})
        self.reference_lora_path = model_config.get("reference_lora_path", None)
        self._kto_dataset_configs = list(config.get("datasets") or [])
        config.pop("datasets", None)
        train_config = config.setdefault("train", {})
        sample_config = config.setdefault("sample", {})
        if not kto_config.dataset_enabled:
            train_config["disable_sampling"] = True
            sample_config["sample_every"] = 0
            sample_config["samples"] = []
        super().__init__(process_id, job, config, **kwargs)
        self.kto_config = kto_config
        if not self.is_ui_trainer and not self.kto_config.dataset_enabled:
            raise ValueError("diffusion_kto_trainer requires the UI SQLite runtime.")

        self._task_counter = 0
        self._kto_root = Path(self.save_root) / "diffusion_kto"
        self._candidate_root = self._kto_root / "candidates"
        self._candidate_root.mkdir(parents=True, exist_ok=True)
        self._offline_examples: list[dict] = []
        self._offline_positive_examples: list[dict] = []
        self._offline_negative_examples: list[dict] = []
        self._offline_positive_index = 0
        self._offline_negative_index = 0
        self._pending_vote_examples: list[dict] = []
        self._pending_vote_task_id: Optional[str] = None
        self._pending_vote_total_steps = 0
        self._pending_vote_completed_steps = 0
        self._pending_vote_started_at = time.monotonic()
        self.reference_lora_network = None
        if self.kto_config.dataset_enabled:
            self._offline_examples = self._load_offline_examples()
            self._offline_positive_examples = [
                example for example in self._offline_examples if float(example.get("reward", 0.0)) > 0.0
            ]
            self._offline_negative_examples = [
                example for example in self._offline_examples if float(example.get("reward", 0.0)) < 0.0
            ]
            if not self._offline_positive_examples or not self._offline_negative_examples:
                raise ValueError("Diffusion-KTO dataset mode requires at least one image in both pos/ and neg/.")
        else:
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
            (self.job_id, DIFFUSION_KTO_TRAINER_TYPE),
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
            (self.job_id, DIFFUSION_KTO_TRAINER_TYPE),
        )
        return int(rows[0]["count"]) if rows else 0

    def _count_active_vote_tasks(self) -> int:
        rows = self._db_execute(
            """
            SELECT COUNT(*) AS count FROM FlowGRPOVoteTask
            WHERE job_id = ? AND trainer_type = ? AND status IN ('generating', 'open')
            """,
            (self.job_id, DIFFUSION_KTO_TRAINER_TYPE),
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

    @staticmethod
    def _image_extensions() -> tuple[str, ...]:
        return (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    @staticmethod
    def _normalize_caption_ext(caption_ext: Optional[str]) -> str:
        caption_ext = (caption_ext or "txt").strip()
        if not caption_ext:
            caption_ext = "txt"
        return caption_ext if caption_ext.startswith(".") else f".{caption_ext}"

    def _caption_for_image(self, image_path: str, *, caption_ext: str, default_caption: Optional[str]) -> str:
        if caption_ext in {".comfyui", "comfyui"}:
            prompt = get_comfyui_caption_from_png_metadata(image_path)
            return prompt or (default_caption or "")

        caption_path = os.path.splitext(image_path)[0] + caption_ext
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as caption_file:
                return clean_caption(caption_file.read())

        default_path_with_ext = os.path.join(os.path.dirname(image_path), f"default{caption_ext}")
        default_path = os.path.join(os.path.dirname(image_path), "default.txt")
        for path in (default_path_with_ext, default_path):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as caption_file:
                    return clean_caption(caption_file.read())
        return default_caption or ""

    def _load_offline_examples(self) -> list[dict]:
        dataset_configs = self._kto_dataset_configs
        if not dataset_configs:
            raise ValueError("Diffusion-KTO dataset mode requires config.process[0].datasets[0].folder_path.")

        dataset_config = dataset_configs[0]
        dataset_root = dataset_config.get("folder_path") or dataset_config.get("dataset_path")
        if not dataset_root:
            raise ValueError("Diffusion-KTO dataset mode requires the selected dataset folder_path.")

        dataset_root = os.path.abspath(os.path.expanduser(str(dataset_root)))
        pos_dir = os.path.join(dataset_root, "pos")
        neg_dir = os.path.join(dataset_root, "neg")
        if not os.path.isdir(pos_dir) or not os.path.isdir(neg_dir):
            raise ValueError("Diffusion-KTO dataset mode expects selected dataset folder to contain pos/ and neg/ folders.")

        caption_ext = self._normalize_caption_ext(dataset_config.get("caption_ext", "txt"))
        default_caption = dataset_config.get("default_caption")
        num_repeats = max(1, int(dataset_config.get("num_repeats", 1) or 1))
        examples: list[dict] = []
        for folder, reward in ((pos_dir, 1.0), (neg_dir, -1.0)):
            for root, _, files in os.walk(folder):
                for filename in sorted(files):
                    if filename.startswith(".") or not filename.lower().endswith(self._image_extensions()):
                        continue
                    image_path = os.path.join(root, filename)
                    examples.append(
                        {
                            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, image_path)),
                            "image_path": image_path,
                            "prompt": self._caption_for_image(
                                image_path,
                                caption_ext=caption_ext,
                                default_caption=default_caption,
                            ),
                            "negative_prompt": "",
                            "reward": reward,
                        }
                    )
        if not examples:
            raise ValueError("Diffusion-KTO dataset mode found no images in pos/ or neg/.")
        examples = examples * num_repeats
        random.shuffle(examples)
        return examples

    def _next_offline_examples(self) -> list[dict]:
        if not self._offline_examples:
            raise ValueError("Diffusion-KTO dataset mode has no offline examples loaded.")
        batch_size = max(1, int(getattr(self.train_config, "batch_size", 1) or 1))
        examples: list[dict] = []
        for _ in range(batch_size):
            use_positive = random.random() < self.kto_config.positive_ratio
            if use_positive:
                if self._offline_positive_index >= len(self._offline_positive_examples):
                    random.shuffle(self._offline_positive_examples)
                    self._offline_positive_index = 0
                examples.append(self._offline_positive_examples[self._offline_positive_index])
                self._offline_positive_index += 1
            else:
                if self._offline_negative_index >= len(self._offline_negative_examples):
                    random.shuffle(self._offline_negative_examples)
                    self._offline_negative_index = 0
                examples.append(self._offline_negative_examples[self._offline_negative_index])
                self._offline_negative_index += 1
        return examples

    def _next_requested_task(self) -> Optional[sqlite3.Row]:
        rows = self._db_execute(
            """
            SELECT * FROM FlowGRPOVoteTask
            WHERE job_id = ? AND trainer_type = ? AND status = 'requested'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (self.job_id, DIFFUSION_KTO_TRAINER_TYPE),
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
        scheduler = task_row["scheduler"] or self.train_config.noise_scheduler or ""
        task_dir = self._task_dir(task_id)

        self._db_execute_write(
            "UPDATE FlowGRPOVoteTask SET status = 'generating' WHERE id = ? AND job_id = ?",
            (task_id, self.job_id),
        )

        try:
            self.update_status("running", "Generating Diffusion-KTO candidate group")
            image_configs: list[GenerateImageConfig] = []
            candidate_rows: list[tuple] = []
            for order_index in range(self.kto_config.group_size):
                self.maybe_stop()
                candidate_id = str(uuid.uuid4())
                seed = self._candidate_seed(task_row, order_index)
                image_path = task_dir / f"{order_index:02d}_{candidate_id}.png"
                state_path = task_dir / f"{order_index:02d}_{candidate_id}.pt"
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
                torch.save(
                    {
                        "task_id": task_id,
                        "candidate_id": candidate_id,
                        "prompt": image_config.prompt,
                        "negative_prompt": image_config.negative_prompt,
                        "seed": seed,
                        "sampler": sampler,
                        "scheduler": scheduler,
                    },
                    state_path,
                )
                image_configs.append(image_config)
                candidate_rows.append(
                    (
                        candidate_id,
                        self.job_id,
                        task_id,
                        DIFFUSION_KTO_TRAINER_TYPE,
                        order_index,
                        image_config.prompt,
                        image_config.negative_prompt,
                        seed,
                        guidance_scale,
                        num_inference_steps,
                        sampler,
                        scheduler,
                        str(image_path),
                        str(state_path),
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
              AND EXISTS (
                SELECT 1
                FROM FlowGRPOVote vote
                WHERE vote.vote_task_id = task.id
                  AND vote.processed = 0
              )
            ORDER BY task.created_at ASC
            LIMIT 1
            """,
            (self.job_id, DIFFUSION_KTO_TRAINER_TYPE),
        )
        return rows[0]["id"] if rows else None

    def _load_task_rows(self, task_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row], list[sqlite3.Row]]:
        task_rows = self._db_execute(
            "SELECT * FROM FlowGRPOVoteTask WHERE id = ? AND job_id = ? AND trainer_type = ? LIMIT 1",
            (task_id, self.job_id, DIFFUSION_KTO_TRAINER_TYPE),
        )
        if not task_rows:
            raise ValueError(f"Diffusion-KTO vote task '{task_id}' was not found.")
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

    def _load_reference_lora_baseline(self) -> None:
        if self.reference_lora_path is None or self.reference_lora_network is not None:
            return

        reference_lora_path = os.path.expanduser(str(self.reference_lora_path))
        raw_lora_state_dict = load_file(reference_lora_path)
        lora_state_dict = raw_lora_state_dict
        if hasattr(self.sd, "convert_lora_weights_before_load"):
            lora_state_dict = self.sd.convert_lora_weights_before_load(raw_lora_state_dict)

        uses_peft_format = bool(
            self.model_config.is_flux or self.model_config.is_v3 or self.model_config.is_lumina2 or self.sd.is_transformer
        )
        modules_dim = {}
        modules_alpha = {}
        for key, value in lora_state_dict.items():
            module_name = None
            if ".lora_A." in key:
                module_name = key.split(".lora_A.")[0]
            elif ".lora_down." in key:
                module_name = key.split(".lora_down.")[0]

            if module_name is not None:
                if uses_peft_format:
                    module_name = module_name.replace(".", "$$")
                modules_dim[module_name] = int(value.shape[0])
            elif key.endswith(".alpha"):
                module_name = key[:-len(".alpha")]
                if uses_peft_format:
                    module_name = module_name.replace(".", "$$")
                modules_alpha[module_name] = value

        for module_name, dim in modules_dim.items():
            modules_alpha.setdefault(module_name, dim)

        if not modules_dim:
            raise ValueError("Diffusion-KTO model.reference_lora_path must point to a standard LoRA safetensors file.")

        linear_dim = next(iter(modules_dim.values()))
        reference_network_config = NetworkConfig(
            type="lora",
            linear=linear_dim,
            linear_alpha=linear_dim,
            transformer_only=False,
            network_kwargs={"only_if_contains": list(modules_dim.keys())},
        )
        network_kwargs = dict(reference_network_config.network_kwargs or {})
        if getattr(self.sd, "target_lora_modules", None) is not None:
            network_kwargs["target_lin_modules"] = self.sd.target_lora_modules

        reference_network = LoRASpecialNetwork(
            text_encoder=self.sd.text_encoder,
            unet=self.sd.get_model_to_train(),
            lora_dim=reference_network_config.linear,
            multiplier=1.0,
            alpha=reference_network_config.linear_alpha,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            train_unet=True,
            train_text_encoder=any(key.startswith("lora_te") or "text_encoder" in key for key in lora_state_dict.keys()),
            is_sdxl=self.model_config.is_xl or self.model_config.is_ssd,
            is_v2=self.model_config.is_v2,
            is_v3=self.model_config.is_v3,
            is_pixart=self.model_config.is_pixart,
            is_auraflow=self.model_config.is_auraflow,
            is_flux=self.model_config.is_flux,
            is_lumina2=self.model_config.is_lumina2,
            is_ssd=self.model_config.is_ssd,
            is_vega=self.model_config.is_vega,
            dropout=0.0,
            use_text_encoder_1=self.model_config.use_text_encoder_1,
            use_text_encoder_2=self.model_config.use_text_encoder_2,
            network_config=reference_network_config,
            network_type=reference_network_config.type,
            transformer_only=reference_network_config.transformer_only,
            is_transformer=self.sd.is_transformer,
            base_model=self.sd,
            is_assistant_adapter=True,
            **network_kwargs,
        )
        reference_network.apply_to(
            self.sd.text_encoder,
            self.sd.get_model_to_train(),
            reference_network.train_text_encoder,
            reference_network.train_unet,
        )
        reference_network.force_to(self.device_torch, dtype=self.sd.torch_dtype)
        reference_network._update_torch_multiplier()
        reference_network.load_weights(raw_lora_state_dict)
        reference_network.requires_grad_(False)
        reference_network.eval()
        reference_network.can_merge_in = False
        reference_network.is_active = False
        self.reference_lora_network = reference_network
        print_acc(f"Loaded Diffusion-KTO reference LoRA baseline from {reference_lora_path}")

    def hook_after_model_load(self):
        super().hook_after_model_load()
        self._load_reference_lora_baseline()

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

    def _sample_timestep(self, latents: torch.Tensor) -> torch.Tensor:
        self._setup_train_timesteps(latents)
        batch_size = latents.shape[0]
        max_index = max(1, len(self.sd.noise_scheduler.timesteps) - 1)
        min_step = max(0, int(self.train_config.min_denoising_steps))
        max_step = min(max_index, int(self.train_config.max_denoising_steps), max_index)
        if self.train_config.timestep_type == "one_step" or min_step >= max_step:
            timestep_index = torch.full((batch_size,), min_step, device=self.device_torch, dtype=torch.long)
        else:
            timestep_index = torch.randint(min_step, max_step + 1, (batch_size,), device=self.device_torch, dtype=torch.long)
        return self.sd.noise_scheduler.timesteps[timestep_index.long()]

    def _encode_prompt_batch(
        self,
        prompts: list[str],
        negative_prompts: list[str],
    ) -> tuple[PromptEmbeds, Optional[PromptEmbeds]]:
        conditional = concat_prompt_embeds(
            [self.sd.encode_prompt(prompt, force_all=True) for prompt in prompts]
        ).to(
            self.device_torch,
            dtype=get_torch_dtype(self.train_config.dtype),
        )
        unconditional = None
        if self.train_config.do_cfg or self.train_config.do_random_cfg:
            unconditional = concat_prompt_embeds(
                [self.sd.encode_prompt(negative_prompt, force_all=True) for negative_prompt in negative_prompts]
            ).to(
                self.device_torch,
                dtype=get_torch_dtype(self.train_config.dtype),
            )
        return conditional, unconditional

    def _kto_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        fake_batch = SimpleNamespace(latents=latents, tensor=latents)
        if hasattr(self.sd, "get_loss_target"):
            return self.sd.get_loss_target(noise=noise, batch=fake_batch, timesteps=timesteps).detach()
        if self.sd.prediction_type == "v_prediction":
            return self.sd.noise_scheduler.get_velocity(latents, noise, timesteps).detach()
        if self.sd.is_flow_matching:
            return (noise - latents).detach()
        return noise.detach()

    def _denoising_losses(
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
        reference_lora = self.reference_lora_network if disable_network else None
        if reference_lora is not None:
            reference_lora.is_active = True
        with contextlib.ExitStack() as stack:
            if disable_network:
                stack.enter_context(torch.no_grad())
                stack.enter_context(self._network_disabled())
            try:
                pred = self.predict_noise(
                    noisy_latents=model_latents.to(self.device_torch, dtype=dtype),
                    timesteps=timesteps,
                    conditional_embeds=conditional_embeds,
                    unconditional_embeds=unconditional_embeds,
                    batch=fake_batch,
                    is_primary_pred=not disable_network,
                )
            finally:
                if reference_lora is not None:
                    reference_lora.is_active = False
        losses = F.mse_loss(pred.float(), target.float(), reduction="none")
        return losses.mean(dim=tuple(range(1, losses.ndim)))

    @staticmethod
    def _roll_tensor_batch(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            rolled = list(value[-1:]) + list(value[:-1])
            return tuple(rolled) if isinstance(value, tuple) else rolled
        return torch.roll(value, shifts=1, dims=0)

    def _roll_prompt_embeds(self, prompt_embeds: Optional[PromptEmbeds]) -> Optional[PromptEmbeds]:
        if prompt_embeds is None:
            return None
        rolled = prompt_embeds.clone()
        if hasattr(rolled, "keys") and callable(rolled.keys):
            for key in rolled.keys():
                rolled[key] = self._roll_tensor_batch(rolled[key])
            return rolled
        if hasattr(rolled, "text_embeds"):
            rolled.text_embeds = self._roll_tensor_batch(rolled.text_embeds)
        if hasattr(rolled, "pooled_embeds"):
            rolled.pooled_embeds = self._roll_tensor_batch(rolled.pooled_embeds)
        if hasattr(rolled, "attention_mask"):
            rolled.attention_mask = self._roll_tensor_batch(rolled.attention_mask)
        return rolled

    def _kl_offset(
        self,
        g_term: torch.Tensor,
        labels_binary: torch.Tensor,
        *,
        model_losses_roll: Optional[torch.Tensor] = None,
        ref_loss_roll: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.kto_config.bce_offset == "sigmoid":
            positive_kl_gpu = (labels_binary * g_term).sum().detach() / (labels_binary.sum() + 1e-6)
            negative_kl_gpu = ((~labels_binary) * g_term).sum().detach() / ((~labels_binary).sum() + 1e-6)
            positive_kl = self.accelerator.reduce(positive_kl_gpu, reduction="mean")
            negative_kl = self.accelerator.reduce(negative_kl_gpu, reduction="mean")
            return ((positive_kl + negative_kl) / 2.0).detach()
        if self.kto_config.bce_offset == "original":
            if model_losses_roll is None or ref_loss_roll is None:
                raise ValueError("Diffusion-KTO bce_offset='original' requires rolled model and reference losses.")
            kl_gpu = torch.mean(ref_loss_roll.detach() - model_losses_roll.detach()).detach()
            return self.accelerator.reduce(kl_gpu, reduction="mean").detach()
        if self.kto_config.bce_offset == "none":
            kl = self.accelerator.reduce(g_term.mean().detach(), reduction="mean")
            return kl.clamp(min=0).detach()
        raise ValueError(f"Unsupported Diffusion-KTO bce_offset '{self.kto_config.bce_offset}'.")

    def _process_kto_examples(self, examples: list[dict]) -> Optional[dict[str, float]]:
        train_rows = [example for example in examples if float(example.get("reward", 0.0)) != 0.0]
        if not train_rows:
            return None
        if self.network is None:
            raise ValueError("Diffusion-KTO trainer requires a trainable LoRA target.")

        dtype = get_torch_dtype(self.train_config.dtype)
        self.network.train()
        self.network.is_active = True

        latents = torch.cat([self._load_image_latents(row["image_path"]) for row in train_rows], dim=0).to(
            self.device_torch,
            dtype=dtype,
        )
        noise = self.sd.get_latent_noise_from_latents(
            latents.to(self.device_torch, dtype=dtype),
            noise_offset=self.train_config.noise_offset,
        ).to(self.device_torch, dtype=dtype)
        timesteps = self._sample_timestep(latents)
        target = self._kto_target(latents, noise, timesteps)
        conditional_embeds, unconditional_embeds = self._encode_prompt_batch(
            [row.get("prompt") or "" for row in train_rows],
            [row.get("negative_prompt") or "" for row in train_rows],
        )

        model_losses = self._denoising_losses(
            latents=latents,
            noise=noise,
            timesteps=timesteps,
            target=target,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            disable_network=False,
        )
        model_losses_roll = None
        ref_loss_roll = None
        if self.kto_config.bce_offset == "original":
            rolled_conditional_embeds = self._roll_prompt_embeds(conditional_embeds)
            rolled_unconditional_embeds = self._roll_prompt_embeds(unconditional_embeds)
            with torch.no_grad():
                model_losses_roll = self._denoising_losses(
                    latents=latents,
                    noise=noise,
                    timesteps=timesteps,
                    target=target,
                    conditional_embeds=rolled_conditional_embeds,
                    unconditional_embeds=rolled_unconditional_embeds,
                    disable_network=False,
                )
        ref_loss = self._denoising_losses(
            latents=latents,
            noise=noise,
            timesteps=timesteps,
            target=target,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            disable_network=True,
        )
        if self.kto_config.bce_offset == "original":
            ref_loss_roll = self._denoising_losses(
                latents=latents,
                noise=noise,
                timesteps=timesteps,
                target=target,
                conditional_embeds=rolled_conditional_embeds,
                unconditional_embeds=rolled_unconditional_embeds,
                disable_network=True,
            )

        policy_kl_logps = -model_losses
        reference_kl_logps = -ref_loss
        g_term = policy_kl_logps - reference_kl_logps
        labels = torch.as_tensor(
            [1.0 if float(row["reward"]) > 0.0 else 0.0 for row in train_rows],
            device=self.device_torch,
            dtype=g_term.dtype,
        )
        labels_binary = labels == 1
        kl = self._kl_offset(
            g_term,
            labels_binary,
            model_losses_roll=model_losses_roll,
            ref_loss_roll=ref_loss_roll,
        )
        g_term = g_term - kl
        label_sgn = (2.0 * labels) - 1.0
        label_scale_g = label_sgn * float(self.kto_config.beta) * g_term
        h = torch.sigmoid(label_scale_g)
        w_y = (
            float(self.kto_config.lambda_d) * labels_binary
            + float(self.kto_config.lambda_u) * (~labels_binary)
        )
        kto_loss = w_y * (1.0 - h)
        loss = kto_loss.mean()
        kto_acc = (label_scale_g > 0).float().mean()
        diff_pos = g_term[labels_binary].sum() / (labels_binary.sum() + 1e-6)
        diff_neg = g_term[~labels_binary].sum() / ((~labels_binary).sum() + 1e-6)

        self.accelerator.backward(loss)
        if not self.is_grad_accumulation_step:
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

        return {
            "loss": float(loss.detach().cpu().item()),
            "kto_loss": float(kto_loss.detach().mean().cpu().item()),
            "kl": float(kl.detach().cpu().item()),
            "kto_acc": float(kto_acc.detach().cpu().item()),
            "model_mse": float(model_losses.detach().mean().cpu().item()),
            "ref_mse": float(ref_loss.detach().mean().cpu().item()),
            "diff_pos": float(diff_pos.detach().cpu().item()),
            "diff_neg": float(diff_neg.detach().cpu().item()),
        }

    def _enqueue_voted_task_examples(self, task_id: str) -> None:
        task_row, candidate_rows, vote_rows = self._load_task_rows(task_id)
        reward_by_candidate = {row["candidate_id"]: float(row["reward"]) for row in vote_rows}
        examples = [
            {
                "id": row["id"],
                "image_path": row["image_path"],
                "prompt": row["prompt"] or task_row["prompt"] or "",
                "negative_prompt": row["negative_prompt"] or task_row["negative_prompt"] or "",
                "reward": reward_by_candidate[row["id"]],
            }
            for row in candidate_rows
            if row["id"] in reward_by_candidate and reward_by_candidate[row["id"]] != 0.0
        ]
        if not examples:
            self._mark_task_processed(task_id)
            return
        positive_examples = [example for example in examples if float(example.get("reward", 0.0)) > 0.0]
        negative_examples = [example for example in examples if float(example.get("reward", 0.0)) < 0.0]
        positive_ratio = float(self.kto_config.positive_ratio)
        if positive_examples and negative_examples and 0.0 < positive_ratio < 1.0:
            positive_count = len(positive_examples)
            negative_count = len(negative_examples)
            target_positive_count = max(
                positive_count,
                math.ceil(negative_count * positive_ratio / (1.0 - positive_ratio)),
            )
            target_negative_count = max(
                negative_count,
                math.ceil(positive_count * (1.0 - positive_ratio) / positive_ratio),
            )
            examples = [dict(positive_examples[index % positive_count]) for index in range(target_positive_count)]
            examples += [dict(negative_examples[index % negative_count]) for index in range(target_negative_count)]
            random.shuffle(examples)
        elif positive_ratio <= 0.0:
            examples = negative_examples
        elif positive_ratio >= 1.0:
            examples = positive_examples
        self._pending_vote_examples.extend(examples)
        batch_size = max(1, int(getattr(self.train_config, "batch_size", 1) or 1))
        self._pending_vote_task_id = task_id
        self._pending_vote_total_steps = math.ceil(len(self._pending_vote_examples) / batch_size)
        self._pending_vote_completed_steps = 0
        self._pending_vote_started_at = time.monotonic()
        self._db_execute_write(
            "UPDATE FlowGRPOVoteTask SET error = ? WHERE id = ? AND job_id = ?",
            (self._kto_apply_progress_payload(0), task_id, self.job_id),
        )

    def _kto_apply_progress_payload(self, completed_steps: int) -> str:
        total_steps = max(1, self._pending_vote_total_steps)
        elapsed = max(1e-6, time.monotonic() - self._pending_vote_started_at)
        remaining_steps = max(0, total_steps - completed_steps)
        it_per_sec = completed_steps / elapsed
        remaining_sec = (remaining_steps / it_per_sec) if it_per_sec > 0 else None
        return json.dumps(
            {
                "type": "grpo_progress",
                "phase": "apply_vote",
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "it_per_sec": it_per_sec,
                "elapsed_sec": elapsed,
                "remaining_sec": remaining_sec,
            }
        )

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        if self.kto_config.dataset_enabled:
            self.update_status("running", "Loaded Diffusion-KTO dataset")
            return
        self._recover_stale_generating_tasks()
        self._promote_requested_vote_tasks()

    def hook_train_loop(self, batch):
        if self.kto_config.dataset_enabled:
            self.update_status("running", "Applying Diffusion-KTO dataset batch")
            return self._process_kto_examples(self._next_offline_examples())

        while True:
            self.maybe_stop()
            self._promote_requested_vote_tasks()
            batch_size = max(1, int(getattr(self.train_config, "batch_size", 1) or 1))
            pending_examples = self._pending_vote_examples[:batch_size]
            if pending_examples:
                self.update_status("running", "Applying Diffusion-KTO candidate vote")
                metrics = self._process_kto_examples(pending_examples)
                if metrics is not None:
                    del self._pending_vote_examples[:len(pending_examples)]
                    self._pending_vote_completed_steps += 1
                    if self._pending_vote_task_id is not None:
                        self._db_execute_write(
                            "UPDATE FlowGRPOVoteTask SET error = ? WHERE id = ? AND job_id = ?",
                            (
                                self._kto_apply_progress_payload(self._pending_vote_completed_steps),
                                self._pending_vote_task_id,
                                self.job_id,
                            ),
                        )
                    if not self._pending_vote_examples and self._pending_vote_task_id is not None:
                        self._mark_task_processed(self._pending_vote_task_id)
                        self._pending_vote_task_id = None
                    self.update_status("running", "Waiting for Diffusion-KTO group votes")
                    return metrics
                continue

            task_id = self._next_voted_task()
            if task_id is None:
                self.update_status("running", "Waiting for Diffusion-KTO group votes")
                time.sleep(2.0)
                continue

            self.update_status("running", "Queueing Diffusion-KTO group votes")
            self._enqueue_voted_task_examples(task_id)

    def get_training_info(self):
        info = super().get_training_info()
        if self.kto_config.dataset_enabled:
            info["diffusion_kto_dataset_examples"] = len(self._offline_examples)
        else:
            info["diffusion_kto_open_tasks"] = self._count_open_tasks()
        return info
