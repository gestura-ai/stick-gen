"""
Async data generation processor with resource management.

Provides concurrent sample generation without overwhelming system resources
through semaphores, worker pools, and backpressure mechanisms.
"""

import asyncio
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import psutil


@dataclass
class ResourceLimits:
    """Configuration for resource limits during async processing."""

    # Concurrency limits
    max_concurrent_samples: int = 4  # Max samples being generated concurrently
    max_concurrent_io: int = 8  # Max concurrent I/O operations
    max_concurrent_llm: int = 2  # Max concurrent LLM API calls

    # Resource thresholds (0.0 - 1.0)
    max_memory_percent: float = 0.80  # Pause if memory exceeds this
    max_cpu_percent: float = 0.90  # Throttle if CPU exceeds this

    # Queue sizes
    write_queue_size: int = 100  # Max samples waiting to be written
    generation_queue_size: int = 50  # Max scenes waiting to generate motion

    # Backpressure settings
    backpressure_delay: float = 0.1  # Seconds to wait when under pressure
    resource_check_interval: float = 1.0  # How often to check resources

    @classmethod
    def from_system(cls, aggressive: bool = False) -> "ResourceLimits":
        """Create limits based on system capabilities."""
        cpu_count = os.cpu_count() or 4
        memory_gb = psutil.virtual_memory().total / (1024**3)

        if aggressive:
            return cls(
                max_concurrent_samples=max(2, cpu_count - 1),
                max_concurrent_io=cpu_count * 2,
                max_concurrent_llm=4,
                max_memory_percent=0.85,
                max_cpu_percent=0.95,
            )
        else:
            # Conservative defaults
            return cls(
                max_concurrent_samples=max(2, cpu_count // 2),
                max_concurrent_io=cpu_count,
                max_concurrent_llm=2,
                max_memory_percent=0.75,
                max_cpu_percent=0.85,
            )


class ResourceMonitor:
    """Monitors system resources and provides backpressure signals."""

    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self._last_check = 0.0
        self._cached_status = {"memory_ok": True, "cpu_ok": True}
        self._lock = threading.Lock()

    def check_resources(self) -> dict[str, bool]:
        """Check if resources are within acceptable limits."""
        now = time.time()

        with self._lock:
            if now - self._last_check < self.limits.resource_check_interval:
                return self._cached_status

            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)

            self._cached_status = {
                "memory_ok": memory.percent / 100 < self.limits.max_memory_percent,
                "cpu_ok": cpu / 100 < self.limits.max_cpu_percent,
                "memory_percent": memory.percent,
                "cpu_percent": cpu,
            }
            self._last_check = now

        return self._cached_status

    async def wait_for_resources(self) -> None:
        """Wait until resources are available."""
        while True:
            status = self.check_resources()
            if status["memory_ok"] and status["cpu_ok"]:
                return
            await asyncio.sleep(self.limits.backpressure_delay)


@dataclass
class AsyncGeneratorStats:
    """Statistics for async generation."""

    samples_generated: int = 0
    samples_written: int = 0
    samples_failed: int = 0
    llm_calls: int = 0
    backpressure_waits: int = 0
    total_time: float = 0.0
    start_time: float = field(default_factory=time.time)

    def samples_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.samples_written / elapsed
        return 0.0


class AsyncSampleWriter:
    """Async writer that batches and writes samples without blocking generation."""

    def __init__(
        self,
        output_dir: str,
        limits: ResourceLimits,
        stats: AsyncGeneratorStats,
        compress: bool = False,
        batch_size: int = 10,
    ):
        self.output_dir = output_dir
        self.limits = limits
        self.compress = compress
        self.batch_size = batch_size
        self._stats = stats  # Shared stats object

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=limits.write_queue_size)
        self._semaphore = asyncio.Semaphore(limits.max_concurrent_io)
        self._shutdown = False
        self._writer_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background writer task."""
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        """Stop the writer and flush remaining samples."""
        self._shutdown = True
        if self._writer_task:
            await self._writer_task

    async def enqueue(self, sample: dict, sample_id: int) -> None:
        """Add a sample to the write queue with backpressure."""
        await self._queue.put((sample, sample_id))

    async def _writer_loop(self) -> None:
        """Background loop that writes samples from queue."""
        batch = []

        while not self._shutdown or not self._queue.empty():
            try:
                sample, sample_id = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
                batch.append((sample, sample_id))

                if len(batch) >= self.batch_size:
                    await self._write_batch(batch)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self._write_batch(batch)
                    batch = []

        # Final flush
        if batch:
            await self._write_batch(batch)

    async def _write_batch(self, batch: list[tuple[dict, int]]) -> None:
        """Write a batch of samples concurrently."""
        tasks = []
        for sample, sample_id in batch:
            tasks.append(self._write_sample(sample, sample_id))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _write_sample(self, sample: dict, sample_id: int) -> None:
        """Write a single sample to disk."""
        async with self._semaphore:
            # Run blocking I/O in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._write_sample_sync, sample, sample_id
            )
            self._stats.samples_written += 1

    def _write_sample_sync(self, sample: dict, sample_id: int) -> None:
        """Synchronous sample writing (runs in thread pool)."""
        import json
        import gzip
        import torch

        filename = f"sample_{sample_id:08d}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Convert tensors to lists for JSON serialization
        json_sample = {}
        for k, v in sample.items():
            if hasattr(v, "tolist"):
                json_sample[k] = v.tolist()
            else:
                json_sample[k] = v

        if self.compress:
            with gzip.open(filepath + ".gz", "wt", encoding="utf-8") as f:
                json.dump(json_sample, f)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_sample, f)


class AsyncDataGenerator:
    """
    Async data generator with resource-aware concurrency.

    Uses semaphores to limit concurrent operations:
    - LLM calls are rate-limited to avoid API throttling
    - Sample generation uses bounded concurrency
    - I/O operations are batched and written asynchronously
    """

    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        output_dir: str = "data/samples",
        compress: bool = False,
    ):
        self.limits = limits or ResourceLimits.from_system()
        self.output_dir = output_dir
        self.compress = compress

        # Semaphores for concurrency control
        self._sample_semaphore = asyncio.Semaphore(self.limits.max_concurrent_samples)
        self._llm_semaphore = asyncio.Semaphore(self.limits.max_concurrent_llm)

        # Resource monitoring
        self._monitor = ResourceMonitor(self.limits)
        self._stats = AsyncGeneratorStats()

        # Writer
        self._writer: Optional[AsyncSampleWriter] = None

        # Thread pool for CPU-bound work (ThreadPoolExecutor is more portable
        # than ProcessPoolExecutor as it doesn't require picklable functions)
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=self.limits.max_concurrent_samples,
            thread_name_prefix="motion_gen",
        )

    async def generate_samples_async(
        self,
        num_samples: int,
        scene_generator: Callable[[], Any],
        motion_generator: Callable[[Any], dict],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AsyncGeneratorStats:
        """
        Generate samples asynchronously with resource management.

        Args:
            num_samples: Number of samples to generate
            scene_generator: Function that generates a scene (may call LLM)
            motion_generator: Function that generates motion from a scene
            progress_callback: Optional callback(completed, total)

        Returns:
            Statistics about the generation run
        """
        os.makedirs(self.output_dir, exist_ok=True)

        self._writer = AsyncSampleWriter(
            self.output_dir, self.limits, self._stats, self.compress
        )
        await self._writer.start()

        try:
            # Create generation tasks
            tasks = []
            for i in range(num_samples):
                task = asyncio.create_task(
                    self._generate_one(
                        i, scene_generator, motion_generator, progress_callback
                    )
                )
                tasks.append(task)

            # Wait for all with concurrency limiting via semaphores
            await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            await self._writer.stop()
            self._cpu_executor.shutdown(wait=True)

        self._stats.total_time = time.time() - self._stats.start_time
        return self._stats

    async def _generate_one(
        self,
        sample_id: int,
        scene_generator: Callable[[], Any],
        motion_generator: Callable[[Any], dict],
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> None:
        """Generate a single sample with resource limits."""
        # Wait for resources
        await self._monitor.wait_for_resources()

        async with self._sample_semaphore:
            try:
                # Generate scene (may involve LLM call)
                scene = await self._generate_scene(scene_generator)

                # Generate motion (CPU-bound)
                loop = asyncio.get_event_loop()
                sample = await loop.run_in_executor(
                    self._cpu_executor, motion_generator, scene
                )

                # Queue for writing
                await self._writer.enqueue(sample, sample_id)
                self._stats.samples_generated += 1

                if progress_callback:
                    progress_callback(self._stats.samples_generated, sample_id + 1)

            except Exception as e:
                self._stats.samples_failed += 1
                print(f"[AsyncGen] Sample {sample_id} failed: {e}")

    async def _generate_scene(self, scene_generator: Callable[[], Any]) -> Any:
        """Generate scene with LLM rate limiting."""
        async with self._llm_semaphore:
            self._stats.llm_calls += 1
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, scene_generator)


async def generate_dataset_async(
    num_samples: int,
    scene_generator: Callable[[], Any],
    motion_generator: Callable[[Any], dict],
    output_dir: str = "data/samples",
    limits: Optional[ResourceLimits] = None,
    compress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> AsyncGeneratorStats:
    """
    Convenience function to generate dataset asynchronously.

    Example:
        ```python
        from src.data_gen.async_processor import generate_dataset_async

        def my_scene_generator():
            return story_engine.generate_random_scene()

        def my_motion_generator(scene):
            # Generate motion tensors from scene
            return {"motion": motion_tensor, "description": scene.description}

        stats = asyncio.run(generate_dataset_async(
            num_samples=1000,
            scene_generator=my_scene_generator,
            motion_generator=my_motion_generator,
            output_dir="data/train_samples",
        ))
        print(f"Generated {stats.samples_written} samples at {stats.samples_per_second():.1f}/sec")
        ```
    """
    generator = AsyncDataGenerator(
        limits=limits,
        output_dir=output_dir,
        compress=compress,
    )

    return await generator.generate_samples_async(
        num_samples=num_samples,
        scene_generator=scene_generator,
        motion_generator=motion_generator,
        progress_callback=progress_callback,
    )

