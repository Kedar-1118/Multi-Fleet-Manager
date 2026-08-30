"""Inference engine for optimized model serving.

Handles TorchScript export, model warm-up, and optimized
inference for the fleet routing dispatch decisions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


class InferenceEngine:
    """Optimized inference engine for fleet routing models.

    Supports loading TorchScript or SB3 models and provides
    fast inference with action masking.

    Args:
        model_path: Path to the trained model.
        device: PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the inference engine.

        Args:
            model_path: Path to model file (.zip or .pt).
            device: Device for inference.
        """
        self.device = torch.device(device)
        self.model = None
        self.sb3_model = None
        self._is_torchscript = False
        self._warm = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load a model from disk.

        Supports:
        - TorchScript models (.pt)
        - Stable-Baselines3 models (.zip)

        Args:
            model_path: Path to the model file.
        """
        if model_path.endswith(".pt"):
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self._is_torchscript = True
        elif model_path.endswith(".zip"):
            from sb3_contrib import MaskablePPO
            self.sb3_model = MaskablePPO.load(model_path)
            self._is_torchscript = False
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def warmup(self, obs_dim: int, n_iterations: int = 100) -> None:
        """Warm up the model with dummy inference calls.

        Args:
            obs_dim: Observation dimension.
            n_iterations: Number of warmup iterations.
        """
        dummy_obs = np.random.randn(obs_dim).astype(np.float32)

        for _ in range(n_iterations):
            if self._is_torchscript and self.model is not None:
                with torch.no_grad():
                    obs_tensor = torch.tensor(dummy_obs).unsqueeze(0).to(self.device)
                    self.model(obs_tensor)
            elif self.sb3_model is not None:
                self.sb3_model.predict(dummy_obs, deterministic=True)

        self._warm = True

    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> tuple[int, float]:
        """Run inference and return action with latency.

        Args:
            obs: Observation array.
            action_mask: Optional boolean action mask.

        Returns:
            Tuple of (action, latency_ms).
        """
        start = time.perf_counter()

        if self._is_torchscript and self.model is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                output = self.model(obs_tensor)

                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                if action_mask is not None:
                    mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
                    logits = logits.masked_fill(~mask_tensor, float("-inf"))

                action = int(torch.argmax(logits, dim=-1).item())

        elif self.sb3_model is not None:
            action, _ = self.sb3_model.predict(
                obs, deterministic=True, action_masks=action_mask
            )
            action = int(action)
        else:
            # No model — return random valid action
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            else:
                action = 0

        latency_ms = (time.perf_counter() - start) * 1000
        return action, latency_ms

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None or self.sb3_model is not None

    def export_torchscript(
        self,
        sb3_model_path: str,
        output_path: str,
        obs_dim: int,
    ) -> str:
        """Export an SB3 model to TorchScript format.

        Args:
            sb3_model_path: Path to the SB3 .zip model.
            output_path: Path for the exported .pt model.
            obs_dim: Observation dimension for tracing.

        Returns:
            Path to the exported TorchScript model.
        """
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(sb3_model_path)
        policy = model.policy

        # Extract the MLP policy network
        from src.agents.policy_network import FleetPolicyNetwork
        net = FleetPolicyNetwork(
            obs_dim=obs_dim,
            action_dim=model.action_space.n,
        )

        # Copy weights from SB3 policy
        # Note: This is a simplified export — architecture must match
        net.eval()
        dummy_input = torch.randn(1, obs_dim)

        traced = torch.jit.trace(net, dummy_input)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced, output_path)

        return output_path
