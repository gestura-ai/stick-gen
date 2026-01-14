"""
Async Data Preparation Pipeline
Gestura AI - https://gestura.ai

Orchestrates asynchronous conversion and processing of all motion capture datasets:
- HumanML3D, KIT-ML, BABEL, BEAT, AMASS, AIST++, NTU-RGB+D, 100STYLE, InterHuman, LSMB19
- Synthetic data generation
- Dataset merging and curation
- Embedding generation
- 2.5D parallax augmentation (Three.js rendering)

Features:
- Parallel conversion of independent datasets
- Resource-aware concurrency control
- Progress tracking with checkpointing
- Resume capability after interruption
- Auto-triggered parallax generation when configured
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import torch


class ConverterStatus(Enum):
    """Status of a dataset converter."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ConverterConfig:
    """Configuration for a single dataset converter."""
    name: str
    module: str  # e.g., "src.data_gen.convert_humanml3d"
    function: str  # e.g., "convert_humanml3d"
    input_dir: str  # Input directory path
    output_path: str  # Output .pt file path
    dependencies: list[str] = field(default_factory=list)  # Other converters this depends on
    required: bool = True  # If False, skip if input doesn't exist
    kwargs: dict[str, Any] = field(default_factory=dict)  # Extra kwargs for converter
    priority: int = 0  # Higher = runs earlier (for independent converters)
    estimated_time_minutes: float = 5.0  # For progress estimation


@dataclass
class PipelineProgress:
    """Tracks progress of the entire pipeline."""
    started_at: str = ""
    converters: dict[str, ConverterStatus] = field(default_factory=dict)
    converter_times: dict[str, float] = field(default_factory=dict)  # Seconds
    errors: dict[str, str] = field(default_factory=dict)
    last_checkpoint: str = ""
    
    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "converters": {k: v.value for k, v in self.converters.items()},
            "converter_times": self.converter_times,
            "errors": self.errors,
            "last_checkpoint": self.last_checkpoint,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PipelineProgress":
        progress = cls()
        progress.started_at = d.get("started_at", "")
        progress.converters = {
            k: ConverterStatus(v) for k, v in d.get("converters", {}).items()
        }
        progress.converter_times = d.get("converter_times", {})
        progress.errors = d.get("errors", {})
        progress.last_checkpoint = d.get("last_checkpoint", "")
        return progress
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> Optional["PipelineProgress"]:
        if os.path.exists(path):
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return None


# Registry of all available converters
CONVERTER_REGISTRY: dict[str, ConverterConfig] = {}


def register_converter(config: ConverterConfig) -> None:
    """Register a converter in the global registry."""
    CONVERTER_REGISTRY[config.name] = config


def _find_data_dir(data_root: str, *candidates: str) -> str:
    """Find the first existing directory from candidates, or return first candidate.

    Also checks for nested directories (e.g., HumanML3D/HumanML3D/).
    """
    for candidate in candidates:
        path = os.path.join(data_root, candidate)
        if os.path.exists(path):
            # Check for nested directory with same name (common pattern)
            nested = os.path.join(path, candidate)
            if os.path.exists(nested):
                return nested
            return path
    # Return first candidate as default (will fail later with clear error)
    return os.path.join(data_root, candidates[0])


def get_default_converters(data_root: str, output_root: str, smpl_path: str) -> list[ConverterConfig]:
    """Get default converter configurations for all supported datasets.

    Supports multiple directory naming conventions for each dataset.
    """
    converters = [
        ConverterConfig(
            name="humanml3d",
            module="src.data_gen.convert_humanml3d",
            function="convert_humanml3d",
            input_dir=_find_data_dir(data_root, "HumanML3D", "humanml3d", "HumanML3d"),
            output_path=os.path.join(output_root, "canonical", "humanml3d.pt"),
            # NOTE: num_workers=1 to avoid nested ProcessPoolExecutor issues when
            # running from ThreadPoolExecutor. The async pipeline already provides
            # concurrency at the converter level.
            kwargs={"num_workers": 1, "include_camera": True},
            priority=10,
            estimated_time_minutes=15,
        ),
        ConverterConfig(
            name="kit_ml",
            module="src.data_gen.convert_kit_ml",
            function="convert_kit_ml",
            input_dir=_find_data_dir(data_root, "KIT-ML", "kit_ml", "KIT_ML", "kit-ml"),
            output_path=os.path.join(output_root, "canonical", "kit_ml.pt"),
            kwargs={},
            priority=10,
            estimated_time_minutes=5,
        ),
        ConverterConfig(
            name="babel",
            module="src.data_gen.convert_babel",
            function="convert_babel",
            input_dir=_find_data_dir(data_root, "amass", "AMASS"),  # amass_root
            output_path=os.path.join(output_root, "canonical", "babel.pt"),
            dependencies=["amass"],  # BABEL uses AMASS motions
            kwargs={
                "babel_path": os.path.join(
                    _find_data_dir(data_root, "babel", "BABEL"), "train.json"
                ),
                "smpl_model_path": smpl_path,
            },
            priority=5,
            estimated_time_minutes=30,
        ),
        ConverterConfig(
            name="amass",
            module="src.data_gen.convert_amass",
            function="convert_amass_dataset",
            input_dir=_find_data_dir(data_root, "amass", "AMASS"),
            output_path=os.path.join(output_root, "canonical", "amass.pt"),
            kwargs={"smpl_model_path": smpl_path},
            priority=10,
            estimated_time_minutes=60,
        ),
        ConverterConfig(
            name="interhuman",
            module="src.data_gen.convert_interhuman",
            function="convert_interhuman",
            input_dir=_find_data_dir(
                data_root, "InterHuman Dataset", "interhuman", "InterHuman", "inter_human"
            ),
            output_path=os.path.join(output_root, "canonical", "interhuman.pt"),
            kwargs={},
            priority=8,
            estimated_time_minutes=20,
        ),
        ConverterConfig(
            name="beat",
            module="src.data_gen.convert_beat",
            function="convert_beat",
            input_dir=_find_data_dir(data_root, "beat", "BEAT"),
            output_path=os.path.join(output_root, "canonical", "beat.pt"),
            required=False,
            priority=7,
            estimated_time_minutes=15,
        ),
        ConverterConfig(
            name="aist_plusplus",
            module="src.data_gen.convert_aist_plusplus",
            function="convert_aist_plusplus",
            input_dir=_find_data_dir(data_root, "aist_plusplus", "AIST++", "aist++"),
            output_path=os.path.join(output_root, "canonical", "aist_plusplus.pt"),
            kwargs={},
            required=False,
            priority=7,
            estimated_time_minutes=25,
        ),
        ConverterConfig(
            name="ntu_rgbd",
            module="src.data_gen.convert_ntu_rgbd",
            function="convert_ntu_rgbd",
            input_dir=_find_data_dir(data_root, "NTU_RGB_D", "ntu_rgbd", "NTU-RGB-D", "ntu-rgbd"),
            output_path=os.path.join(output_root, "canonical", "ntu_rgbd.pt"),
            required=False,
            priority=6,
            estimated_time_minutes=45,
        ),
        ConverterConfig(
            name="100style",
            module="src.data_gen.convert_100style_canonical",
            function="convert_100style_canonical",
            input_dir=_find_data_dir(data_root, "100Style", "100STYLE", "100style"),
            output_path=os.path.join(output_root, "canonical", "100style.pt"),
            required=False,
            priority=8,
            estimated_time_minutes=10,
        ),
        ConverterConfig(
            name="lsmb19",
            module="src.data_gen.convert_lsmb19",
            function="convert_lsmb19",
            input_dir=_find_data_dir(data_root, "lsmb19-mocap", "lsmb19", "LSMB19"),
            output_path=os.path.join(output_root, "canonical", "lsmb19.pt"),
            required=False,
            priority=5,
            estimated_time_minutes=10,
        ),
    ]
    return converters


class AsyncDataPrepPipeline:
    """Async orchestrator for the complete data preparation pipeline."""

    def __init__(
        self,
        converters: list[ConverterConfig],
        max_concurrent: int = 4,
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
        config_path: str = "configs/base.yaml",
    ):
        self.converters = {c.name: c for c in converters}
        self.max_concurrent = max_concurrent
        self.checkpoint_path = checkpoint_path or "data/.pipeline_progress.json"
        self.resume = resume
        self.config_path = config_path
        self.progress = PipelineProgress()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _load_progress(self) -> None:
        """Load progress from checkpoint if resuming."""
        if self.resume:
            loaded = PipelineProgress.load(self.checkpoint_path)
            if loaded:
                self.progress = loaded
                print(f"[Pipeline] Resuming from checkpoint: {self.checkpoint_path}")
                completed = [
                    k for k, v in self.progress.converters.items()
                    if v == ConverterStatus.COMPLETED
                ]
                print(f"[Pipeline] Already completed: {completed}")

    def _save_checkpoint(self) -> None:
        """Save current progress to checkpoint."""
        self.progress.last_checkpoint = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.progress.save(self.checkpoint_path)

    def _can_run(self, name: str) -> bool:
        """Check if a converter can run (dependencies satisfied)."""
        config = self.converters[name]
        for dep in config.dependencies:
            if dep not in self.progress.converters:
                return False
            if self.progress.converters[dep] != ConverterStatus.COMPLETED:
                return False
        return True

    def _should_skip(self, name: str) -> bool:
        """Check if converter should be skipped."""
        config = self.converters[name]
        # Already completed
        if self.progress.converters.get(name) == ConverterStatus.COMPLETED:
            if os.path.exists(config.output_path):
                return True
        # Input doesn't exist and not required
        if not config.required and not os.path.exists(config.input_dir):
            return True
        return False

    async def _run_converter(self, name: str) -> bool:
        """Run a single converter asynchronously."""
        config = self.converters[name]

        if self._should_skip(name):
            self.progress.converters[name] = ConverterStatus.SKIPPED
            print(f"[Pipeline] Skipping {name} (already done or input missing)")
            return True

        async with self._semaphore:
            self.progress.converters[name] = ConverterStatus.RUNNING
            self._save_checkpoint()
            print(f"[Pipeline] Starting {name}...")

            start_time = time.time()
            try:
                # Import and run the converter
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self._run_converter_sync,
                    config,
                )

                elapsed = time.time() - start_time
                self.progress.converter_times[name] = elapsed
                self.progress.converters[name] = ConverterStatus.COMPLETED
                self._save_checkpoint()

                print(f"[Pipeline] ✓ {name} completed in {elapsed:.1f}s")
                return True

            except Exception as e:
                elapsed = time.time() - start_time
                self.progress.converter_times[name] = elapsed
                self.progress.converters[name] = ConverterStatus.FAILED
                self.progress.errors[name] = str(e)
                self._save_checkpoint()

                print(f"[Pipeline] ✗ {name} failed: {e}")
                return False

    def _run_converter_sync(self, config: ConverterConfig) -> Any:
        """Synchronous converter execution (runs in thread pool)."""
        import importlib
        import inspect

        module = importlib.import_module(config.module)
        func = getattr(module, config.function)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

        # Build kwargs from config
        kwargs = dict(config.kwargs)

        # Get actual function parameters (not local variables)
        sig = inspect.signature(func)
        param_names = set(sig.parameters.keys())

        # Map input_dir to the appropriate parameter name
        input_param_names = ["root_dir", "data_root", "amass_root", "style_dir", "root", "input_dir"]
        for param_name in input_param_names:
            if param_name in param_names and param_name not in kwargs:
                kwargs[param_name] = config.input_dir
                break

        # Map output_path to the appropriate parameter name
        if "output_path" in param_names and "output_path" not in kwargs:
            kwargs["output_path"] = config.output_path
        elif "output" in param_names and "output" not in kwargs:
            kwargs["output"] = config.output_path

        return func(**kwargs)

    async def run_all(self) -> dict[str, Any]:
        """Run all converters with dependency ordering."""
        self.progress.started_at = datetime.now().isoformat()
        self._load_progress()

        # Initialize status for all converters
        for name in self.converters:
            if name not in self.progress.converters:
                self.progress.converters[name] = ConverterStatus.PENDING

        # Topological sort by priority and dependencies
        pending = set(self.converters.keys())
        running: set[asyncio.Task] = set()
        completed: set[str] = set()

        # Mark already completed
        for name, status in list(self.progress.converters.items()):
            if status == ConverterStatus.COMPLETED:
                completed.add(name)
                pending.discard(name)

        while pending or running:
            # Find converters ready to run
            ready = []
            for name in list(pending):
                if self._can_run(name):
                    ready.append(name)

            # Sort by priority (higher first)
            ready.sort(key=lambda n: self.converters[n].priority, reverse=True)

            # Start as many as we can
            for name in ready:
                if len(running) >= self.max_concurrent:
                    break
                pending.remove(name)
                task = asyncio.create_task(self._run_converter(name))
                task.converter_name = name  # type: ignore
                running.add(task)

            if running:
                # Wait for at least one to complete
                done, running = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    name = task.converter_name  # type: ignore
                    if task.result():
                        completed.add(name)
            elif pending:
                # No running tasks and can't start any - dependency deadlock?
                blocked = [n for n in pending if not self._can_run(n)]
                print(f"[Pipeline] Warning: Blocked converters: {blocked}")
                break

        # Summary
        stats = {
            "completed": len([
                n for n, s in self.progress.converters.items()
                if s == ConverterStatus.COMPLETED
            ]),
            "failed": len([
                n for n, s in self.progress.converters.items()
                if s == ConverterStatus.FAILED
            ]),
            "skipped": len([
                n for n, s in self.progress.converters.items()
                if s == ConverterStatus.SKIPPED
            ]),
            "total_time": sum(self.progress.converter_times.values()),
            "errors": self.progress.errors,
        }

        return stats

    async def run_synthetic(
        self,
        num_samples: int = 10000,
        output_path: Optional[str] = None,
        motion_source_path: Optional[str] = None,
        use_llm_override: Optional[bool] = None,
    ) -> None:
        """Run synthetic data generation asynchronously.

        Args:
            num_samples: Number of synthetic samples to generate
            output_path: Path for output .pt file
            motion_source_path: Path to merged canonical data to condition on.
                If provided, synthetic samples will be generated by augmenting
                real motion clips from this dataset.
            use_llm_override: Override config LLM setting (None = use config,
                True = force LLM on, False = force LLM off). Passed through from
                CLI --use-llm flag. When enabled, descriptions are generated by
                Grok instead of templates. Requires GROK_API_KEY environment variable.

        LLM Configuration:
            LLM settings are read from config file under `data_generation.llm`.
            The priority is: CLI override > env vars > config file.
            See MotionConditionedSampleGenerator for full override logic.
        """
        from .dataset_generator import generate_dataset_async, merge_samples
        from .async_processor import ResourceLimits

        output = output_path or os.path.join(
            os.path.dirname(self.checkpoint_path).replace("/.pipeline", ""),
            "canonical", "synthetic.pt"
        )
        samples_dir = os.path.dirname(output)

        if motion_source_path and os.path.exists(motion_source_path):
            print(f"[Pipeline] Generating {num_samples} motion-conditioned synthetic samples...")
            print(f"[Pipeline] Motion source: {motion_source_path}")
            print(f"[Pipeline] Config: {self.config_path}")
            if use_llm_override is not None:
                print(f"[Pipeline] LLM CLI override: {'--use-llm (force on)' if use_llm_override else 'disabled'}")
            else:
                print(f"[Pipeline] LLM: using config/env var settings")
        else:
            print(f"[Pipeline] Generating {num_samples} procedural synthetic samples...")
            motion_source_path = None  # Ensure None if file doesn't exist

        limits = ResourceLimits.from_system()
        limits.max_concurrent_llm = 2

        await generate_dataset_async(
            config_path=self.config_path,
            num_samples=num_samples,
            output_dir=samples_dir,
            resource_limits=limits,
            use_llm_override=use_llm_override,
            motion_source_path=motion_source_path,
        )

        # Merge individual JSON sample files into a single .pt dataset
        # This is required for the pipeline merge step to find the synthetic data
        print(f"[Pipeline] Merging JSON samples into {output}...")
        try:
            merge_samples(
                samples_dir=samples_dir,
                output_path=output,
                delete_after_merge=True,  # Clean up JSON files
                verbose=True,
            )
            print(f"[Pipeline] ✓ Synthetic generation complete: {output}")
        except ValueError as e:
            print(f"[Pipeline] ⚠️ Synthetic merge failed: {e}")
            print(f"[Pipeline] JSON files remain in {samples_dir}")

    async def run_merge(
        self,
        output_path: str,
        balance_sources: bool = True,
        max_source_fraction: float = 0.3,
        extra_input_paths: list[str] | None = None,
    ) -> None:
        """Merge all converted datasets."""
        from scripts.merge_datasets import merge_datasets

        # Collect all completed canonical files
        input_paths = []
        for name, config in self.converters.items():
            if self.progress.converters.get(name) == ConverterStatus.COMPLETED:
                if os.path.exists(config.output_path):
                    input_paths.append(config.output_path)

        if extra_input_paths:
            for p in extra_input_paths:
                if os.path.exists(p):
                    input_paths.append(p)

        if not input_paths:
            print("[Pipeline] No datasets to merge!")
            return

        print(f"[Pipeline] Merging {len(input_paths)} datasets...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: merge_datasets(
                input_paths=input_paths,
                output_path=output_path,
                balance_sources=balance_sources,
                max_source_fraction=max_source_fraction,
            ),
        )
        print(f"[Pipeline] ✓ Merged to {output_path}")

    async def run_curation(
        self,
        merged_path: str,
        output_dir: str,
        min_quality_pretrain: float = 0.5,
        min_quality_sft: float = 0.8,
    ) -> None:
        """Run curation to create pretrain/SFT splits."""
        from src.data_gen.curation import CurationConfig, curate_samples

        print(f"[Pipeline] Curating {merged_path}...")
        samples = torch.load(merged_path)

        cfg = CurationConfig(
            min_quality_pretrain=min_quality_pretrain,
            min_quality_sft=min_quality_sft,
        )

        loop = asyncio.get_event_loop()
        pretrain, sft, stats = await loop.run_in_executor(
            self._executor,
            lambda: curate_samples(samples, cfg),
        )

        os.makedirs(output_dir, exist_ok=True)
        torch.save(pretrain, os.path.join(output_dir, "pretrain_data.pt"))
        torch.save(sft, os.path.join(output_dir, "sft_data.pt"))

        with open(os.path.join(output_dir, "curation_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"[Pipeline] ✓ Curated: {len(pretrain)} pretrain, {len(sft)} SFT")

    async def run_parallax(
        self,
        dataset_path: str,
        output_dir: str,
        views_per_motion: int = 250,
        frames_per_view: int = 4,
        node_script: str = "src/data_gen/renderers/threejs_parallax_renderer.js",
        max_samples: int | None = None,
        minimal: bool = True,
    ) -> dict[str, Any]:
        """Generate 2.5D parallax PNG frames using Three.js renderer.

        This is Stage 2 of the data pipeline, running after canonical motion
        data generation but before training.

        Args:
            dataset_path: Path to the curated .pt dataset file
            output_dir: Output directory for parallax PNG frames
            views_per_motion: Number of camera trajectories per actor motion
            frames_per_view: Frames rendered per camera trajectory
            node_script: Path to the Node.js Three.js renderer script
            max_samples: Optional limit on samples to process (for debugging)
            minimal: If True, disable scene props and backgrounds (clean training data)

        Returns:
            Dict with generation stats (samples_processed, frames_generated, etc.)
        """
        from src.data_gen.parallax_augmentation import (
            generate_parallax_for_dataset,
            _has_node_runtime,
        )

        if not _has_node_runtime():
            print("[Pipeline] ⚠️  Node.js not found, skipping parallax generation")
            print("[Pipeline]    Install Node.js and run: npm install -g three gl canvas")
            return {"skipped": True, "reason": "node_not_found"}

        if not os.path.exists(dataset_path):
            print(f"[Pipeline] ⚠️  Dataset not found: {dataset_path}")
            return {"skipped": True, "reason": "dataset_not_found"}

        print(f"[Pipeline] Generating parallax frames...")
        print(f"           Dataset: {dataset_path}")
        print(f"           Output: {output_dir}")
        print(f"           Views/motion: {views_per_motion}, Frames/view: {frames_per_view}")

        loop = asyncio.get_event_loop()
        start_time = time.time()

        await loop.run_in_executor(
            self._executor,
            lambda: generate_parallax_for_dataset(
                dataset_path=dataset_path,
                output_dir=output_dir,
                views_per_motion=views_per_motion,
                node_script=node_script,
                max_samples=max_samples,
                frames_per_view=frames_per_view,
                minimal=minimal,
            ),
        )

        elapsed = time.time() - start_time
        print(f"[Pipeline] ✓ Parallax generation complete ({elapsed / 60:.1f} min)")

        return {
            "skipped": False,
            "elapsed_seconds": elapsed,
            "output_dir": output_dir,
        }


async def run_full_pipeline(
    data_root: str,
    output_root: str,
    smpl_path: str = "data/smpl_models",
    max_concurrent: int = 4,
    synthetic_samples: int = 10000,
    resume: bool = True,
    skip_synthetic: bool = False,
    use_llm_override: Optional[bool] = None,
    config_path: str = "configs/base.yaml",
    skip_parallax: bool = False,
    only_parallax: bool = False,
) -> dict[str, Any]:
    """Run the complete data preparation pipeline asynchronously.

    Pipeline Stages:
        1. Convert datasets to canonical format
        2. Merge canonical datasets
        3. Generate motion-conditioned synthetic data
        4. Curate datasets (pretrain/SFT split)
        5. Generate 2.5D parallax frames (if enabled in config)

    Args:
        data_root: Root directory containing raw datasets
        output_root: Output directory for processed data
        smpl_path: Path to SMPL model files
        max_concurrent: Max concurrent converter processes
        synthetic_samples: Number of synthetic samples to generate
        resume: Whether to resume from checkpoint
        skip_synthetic: Skip synthetic data generation
        use_llm_override: Override config LLM setting (None = use config,
            True = force LLM on, False = force LLM off). For CLI --use-llm flag.
        config_path: Path to YAML configuration file
        skip_parallax: Skip parallax generation even if enabled in config
        only_parallax: Run ONLY parallax generation, skipping phases 1-4.
            Use when motion data already exists and you just need parallax frames.

    LLM Configuration:
        LLM settings for motion descriptions are read from config file under
        `data_generation.llm`. The priority is: CLI override > env vars > config file.
        See MotionConditionedSampleGenerator for full override logic.

    Parallax Configuration:
        Parallax settings are read from config file under `data_generation.parallax`.
        When `enabled: true`, parallax generation runs automatically after curation.
    """
    # Load config for parallax settings
    import yaml
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    parallax_config = config.get("data_generation", {}).get("parallax", {})
    # Force parallax enabled if --only-parallax, otherwise check config
    parallax_enabled = only_parallax or (
        parallax_config.get("enabled", False) and not skip_parallax
    )

    # Basic config validation for multi-frame sequence training.
    # When sequence mode is enabled (`data.use_parallax_sequences: true`), we
    # expect each camera trajectory to have at least as many frames as the
    # requested training sequence length. If this is not true, the
    # MultimodalParallaxSequenceDataset may end up with zero sequences or
    # heavily truncated windows.
    data_cfg = config.get("data", {})
    seq_len = data_cfg.get("parallax_sequence_length")
    use_sequences = data_cfg.get("use_parallax_sequences", False)
    frames_per_view = parallax_config.get("frames_per_view")

    if (
        parallax_enabled
        and use_sequences
        and isinstance(seq_len, int)
        and isinstance(frames_per_view, int)
        and frames_per_view < seq_len
    ):
        print(
            "[Pipeline] ⚠️  Configuration mismatch: "
            "data_generation.parallax.frames_per_view "
            f"({frames_per_view}) < data.parallax_sequence_length ({seq_len})."
        )
        print(
            "[Pipeline]      The sequence dataset may be empty or truncated. "
            "Increase frames_per_view or reduce parallax_sequence_length."
        )

    parallax_root = data_cfg.get("parallax_root", "data/2.5d_parallax")
    curated_dir = os.path.join(output_root, "curated")

    print("=" * 60)
    if only_parallax:
        print("Stick-Gen Parallax Generation (Phase 5 Only)")
    else:
        print("Stick-Gen Async Data Preparation Pipeline")
    print("=" * 60)
    print(f"Config: {config_path}")
    if not only_parallax:
        print(f"Data root: {data_root}")
        print(f"Output root: {output_root}")
        print(f"Max concurrent: {max_concurrent}")
        print(f"Synthetic samples: {synthetic_samples}")
        if use_llm_override is not None:
            print(f"LLM CLI override: {'--use-llm (force on)' if use_llm_override else 'disabled'}")
        else:
            print("LLM: using config/env var settings")
    print(f"Parallax: {'enabled' if parallax_enabled else 'disabled'}")
    print(f"Parallax output: {parallax_root}")
    print()

    start_time = time.time()
    converter_stats: dict[str, Any] = {"completed": 0, "failed": 0, "skipped": 0}
    parallax_stats: dict[str, Any] = {"skipped": True, "reason": "disabled"}

    # --only-parallax: Skip phases 1-4, jump directly to phase 5
    if only_parallax:
        print("[Phase 5/5] Generating 2.5D parallax frames (--only-parallax mode)...")
        curated_pretrain = os.path.join(curated_dir, "pretrain_data.pt")
        if not os.path.exists(curated_pretrain):
            print(f"ERROR: Curated dataset not found: {curated_pretrain}")
            print("       Run full pipeline first (without --only-parallax)")
            return {"error": "curated_dataset_not_found", "path": curated_pretrain}

        # Create minimal pipeline for parallax only
        pipeline = AsyncDataPrepPipeline(
            converters=[],
            max_concurrent=max_concurrent,
            checkpoint_path=os.path.join(output_root, ".pipeline_progress.json"),
            resume=resume,
            config_path=config_path,
        )
        parallax_stats = await pipeline.run_parallax(
            dataset_path=curated_pretrain,
            output_dir=parallax_root,
            views_per_motion=parallax_config.get("views_per_motion", 250),
            frames_per_view=parallax_config.get("frames_per_view", 4),
            node_script=parallax_config.get(
                "node_script", "src/data_gen/renderers/threejs_parallax_renderer.js"
            ),
            max_samples=parallax_config.get("max_samples"),
            minimal=parallax_config.get("minimal", True),
        )
    else:
        # Full pipeline: phases 1-5
        converters = get_default_converters(data_root, output_root, smpl_path)
        pipeline = AsyncDataPrepPipeline(
            converters=converters,
            max_concurrent=max_concurrent,
            checkpoint_path=os.path.join(output_root, ".pipeline_progress.json"),
            resume=resume,
            config_path=config_path,
        )

        total_phases = 5 if parallax_enabled else 4

        # Phase 1: Convert all datasets to canonical format
        print(f"\n[Phase 1/{total_phases}] Converting datasets...")
        converter_stats = await pipeline.run_all()

        # Phase 2: Merge canonical datasets (BEFORE synthetic generation)
        print(f"\n[Phase 2/{total_phases}] Merging canonical datasets...")
        merged_path = os.path.join(output_root, "merged_canonical.pt")
        await pipeline.run_merge(merged_path)

        # Phase 3: Synthetic generation CONDITIONED on merged motion data
        synthetic_path = os.path.join(output_root, "canonical", "synthetic.pt")
        
        if not skip_synthetic:
            print(f"\n[Phase 3/{total_phases}] Generating motion-conditioned synthetic data...")
            await pipeline.run_synthetic(
                num_samples=synthetic_samples,
                output_path=synthetic_path,
                motion_source_path=merged_path,  # Condition on merged data
                use_llm_override=use_llm_override,
            )
        else:
            print(f"\n[Phase 3/{total_phases}] Skipping synthetic generation (--skip-synthetic)...")

        # Phase 3b: Re-merge to include synthetic data (if it exists)
        if os.path.exists(synthetic_path):
            print(f"[Pipeline] Merging synthetic data from {synthetic_path}...")
            await pipeline.run_merge(merged_path, extra_input_paths=[synthetic_path])
        else:
            print("[Pipeline] No synthetic data found to merge.")

        # Phase 4: Curation
        print(f"\n[Phase 4/{total_phases}] Curating datasets...")
        await pipeline.run_curation(merged_path, curated_dir)

        # Phase 5: Parallax generation (if enabled)
        if parallax_enabled:
            print(f"\n[Phase 5/{total_phases}] Generating 2.5D parallax frames...")
            curated_pretrain = os.path.join(curated_dir, "pretrain_data.pt")
            parallax_stats = await pipeline.run_parallax(
                dataset_path=curated_pretrain,
                output_dir=parallax_root,
                views_per_motion=parallax_config.get("views_per_motion", 250),
                frames_per_view=parallax_config.get("frames_per_view", 4),
                node_script=parallax_config.get(
                    "node_script", "src/data_gen/renderers/threejs_parallax_renderer.js"
                ),
                max_samples=parallax_config.get("max_samples"),
                minimal=parallax_config.get("minimal", True),
            )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Converters: {converter_stats['completed']} completed, "
          f"{converter_stats['failed']} failed, "
          f"{converter_stats['skipped']} skipped")
    print(f"Output: {curated_dir}")
    if parallax_enabled and not parallax_stats.get("skipped"):
        print(f"Parallax: {parallax_stats.get('output_dir', parallax_root)}")

    return {
        "elapsed_seconds": elapsed,
        "converters": converter_stats,
        "parallax": parallax_stats,
    }


def main():
    """CLI entrypoint for async data preparation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Async data preparation pipeline for Stick-Gen"
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root directory containing raw datasets",
    )
    parser.add_argument(
        "--output-root", type=str, default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--smpl-path", type=str, default="data/smpl_models",
        help="Path to SMPL model files",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=4,
        help="Max concurrent converter processes",
    )
    parser.add_argument(
        "--synthetic-samples", type=int, default=10000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip synthetic data generation",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Override config to force Grok LLM for descriptions (requires GROK_API_KEY). "
             "Without this flag, LLM setting is read from config file (data_generation.llm.use_mock).",
    )
    parser.add_argument(
        "--skip-parallax", action="store_true",
        help="Skip 2.5D parallax frame generation even if enabled in config. "
             "Parallax is auto-triggered when data_generation.parallax.enabled: true.",
    )
    parser.add_argument(
        "--only-parallax", action="store_true",
        help="Run ONLY parallax generation (Phase 5), skipping phases 1-4. "
             "Use when motion data is already generated and you just need parallax frames.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming from checkpoint",
    )
    args = parser.parse_args()

    # Convert --use-llm flag to override (None means use config, True means force on)
    use_llm_override = True if args.use_llm else None

    asyncio.run(run_full_pipeline(
        data_root=args.data_root,
        output_root=args.output_root,
        smpl_path=args.smpl_path,
        max_concurrent=args.max_concurrent,
        synthetic_samples=args.synthetic_samples,
        resume=not args.no_resume,
        skip_synthetic=args.skip_synthetic,
        use_llm_override=use_llm_override,
        config_path=args.config,
        skip_parallax=args.skip_parallax,
        only_parallax=args.only_parallax,
    ))


if __name__ == "__main__":
    main()

