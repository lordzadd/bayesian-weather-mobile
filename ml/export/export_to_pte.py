"""
Exports the trained BMA model to ExecuTorch .pte format for on-device inference.

Steps:
  1. Load trained PyTorch checkpoint
  2. Trace the predict() method with torch.export
  3. Apply ExecuTorch edge compilation and Vulkan/Metal delegation
  4. Serialize to .pte

Requirements:
    pip install executorch
"""

import logging
from pathlib import Path

import torch
import pyro

from training.bma_model import BMAModel

try:
    from executorch.exir import to_edge, EdgeCompileConfig
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False
    logging.warning("ExecuTorch not installed. Run: pip install executorch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
EXPORT_DIR = Path(__file__).parent.parent / "export"
EXPORT_DIR.mkdir(exist_ok=True)

N_FEATURES = 6
BATCH_SIZE = 1  # Single-sample inference on device


def load_model(checkpoint_path: Path) -> BMAModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = BMAModel(n_features=N_FEATURES)
    model.load_state_dict(ckpt["model_state"])
    pyro.get_param_store().set_state(ckpt["pyro_params"])
    model.eval()
    return model


class BMAInferenceModule(torch.nn.Module):
    """
    Thin wrapper exposing only predict() for ExecuTorch tracing.
    Fixes n_samples at export time for a static compute graph.
    """

    def __init__(self, bma: BMAModel, n_samples: int = 50):
        super().__init__()
        self.bma = bma
        self.n_samples = n_samples

    def forward(self, gfs_forecast: torch.Tensor, spatial_embed: torch.Tensor):
        return self.bma.predict(gfs_forecast, spatial_embed, n_samples=self.n_samples)


def export_model(checkpoint_path: Path, output_path: Path, n_samples: int = 50):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    bma = load_model(checkpoint_path)

    wrapper = BMAInferenceModule(bma, n_samples=n_samples)
    wrapper.eval()

    # Example inputs matching single-sample on-device inference
    gfs_example = torch.zeros(BATCH_SIZE, N_FEATURES)
    spatial_example = torch.zeros(BATCH_SIZE, 2)
    example_inputs = (gfs_example, spatial_example)

    logger.info("Tracing model with torch.export...")
    exported = torch.export.export(wrapper, example_inputs)

    if not HAS_EXECUTORCH:
        # Fallback: save TorchScript for CPU SIMD path
        scripted = torch.jit.script(wrapper)
        ts_path = output_path.with_suffix(".pt")
        scripted.save(str(ts_path))
        logger.info(f"ExecuTorch unavailable. Saved TorchScript to {ts_path}")
        return

    logger.info("Compiling to ExecuTorch edge IR...")
    edge_program = to_edge(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=True),
    )

    # Attempt GPU delegation (Vulkan on Android, Metal on iOS handled at runtime)
    try:
        edge_program = edge_program.to_backend(VulkanPartitioner())
        logger.info("Vulkan GPU partitioner applied.")
    except Exception as e:
        logger.warning(f"GPU delegation skipped: {e}. Falling back to CPU.")

    exec_prog = edge_program.to_executorch()

    pte_path = str(output_path)
    with open(pte_path, "wb") as f:
        exec_prog.write_to_file(f)
    logger.info(f"Exported .pte model to {pte_path}")


if __name__ == "__main__":
    ckpt = CHECKPOINT_DIR / "bma_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError("No checkpoint found. Run ml/training/train.py first.")

    out = EXPORT_DIR / "bma_model.pte"
    export_model(ckpt, out, n_samples=50)
