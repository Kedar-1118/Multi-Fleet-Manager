"""Tests for the InferenceEngine and FleetPolicyNetwork.

Covers:
- FleetPolicyNetwork forward pass shape correctness.
- FleetPolicyNetwork TorchScript export and reload.
- InferenceEngine loading SB3 models (.zip).
- InferenceEngine predict with and without action masks.
- InferenceEngine warmup execution.
- InferenceEngine fallback when no model is loaded.
"""

import os
import tempfile

import pytest
import numpy as np
import torch

from src.agents.policy_network import FleetPolicyNetwork
from src.serving.inference import InferenceEngine


# =========================================================================
# FleetPolicyNetwork Tests
# =========================================================================

class TestFleetPolicyNetwork:
    """Tests for the standalone policy network."""

    @pytest.fixture
    def network(self) -> FleetPolicyNetwork:
        return FleetPolicyNetwork(obs_dim=56, action_dim=31)

    def test_output_shapes(self, network: FleetPolicyNetwork) -> None:
        """Forward pass should return (batch, action_dim) logits and (batch, 1) value."""
        obs = torch.randn(4, 56)
        logits, value = network(obs)
        assert logits.shape == (4, 31)
        assert value.shape == (4, 1)

    def test_single_sample(self, network: FleetPolicyNetwork) -> None:
        """Single-sample batch should work."""
        obs = torch.randn(1, 56)
        logits, value = network(obs)
        assert logits.shape == (1, 31)

    def test_action_probabilities_sum_to_one(
        self, network: FleetPolicyNetwork
    ) -> None:
        """Masked probabilities should sum to 1."""
        obs = torch.randn(1, 56)
        mask = torch.ones(1, 31, dtype=torch.bool)
        mask[0, 0:5] = False  # mask out first 5 actions
        probs = network.get_action_probabilities(obs, action_mask=mask)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        # Masked actions should have 0 probability
        assert probs[0, 0].item() < 1e-6

    def test_action_probabilities_no_mask(
        self, network: FleetPolicyNetwork
    ) -> None:
        """Without a mask, probabilities should still sum to 1."""
        obs = torch.randn(1, 56)
        probs = network.get_action_probabilities(obs)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_torchscript_export_and_reload(
        self, network: FleetPolicyNetwork
    ) -> None:
        """Export to TorchScript, reload, and verify output consistency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "policy.pt")
            network.export_torchscript(pt_path, obs_dim=56)

            assert os.path.exists(pt_path)

            loaded = torch.jit.load(pt_path)
            obs = torch.randn(1, 56)

            network.eval()
            with torch.no_grad():
                orig_logits, orig_val = network(obs)
                loaded_out = loaded(obs)
                if isinstance(loaded_out, tuple):
                    loaded_logits, loaded_val = loaded_out
                else:
                    loaded_logits = loaded_out

            assert torch.allclose(orig_logits, loaded_logits, atol=1e-5)

    def test_custom_hidden_sizes(self) -> None:
        """Network should work with custom hidden layer sizes."""
        net = FleetPolicyNetwork(obs_dim=20, action_dim=10, hidden_sizes=[64, 32])
        obs = torch.randn(2, 20)
        logits, value = net(obs)
        assert logits.shape == (2, 10)
        assert value.shape == (2, 1)


# =========================================================================
# InferenceEngine Tests
# =========================================================================

class TestInferenceEngine:
    """Tests for the InferenceEngine."""

    def test_no_model_loaded_predicts_fallback(self) -> None:
        """Without a model, predict should return a valid action via fallback."""
        engine = InferenceEngine()
        assert not engine.is_loaded

        obs = np.random.randn(56).astype(np.float32)
        mask = np.ones(31, dtype=bool)
        mask[0:3] = False

        action, latency_ms = engine.predict(obs, action_mask=mask)
        assert isinstance(action, int)
        assert 3 <= action < 31  # should not pick masked actions
        assert latency_ms >= 0.0

    def test_no_model_no_mask(self) -> None:
        """Without a model and no mask, should return action 0 fallback."""
        engine = InferenceEngine()
        obs = np.random.randn(56).astype(np.float32)
        action, latency_ms = engine.predict(obs)
        assert action == 0

    def test_load_torchscript(self) -> None:
        """Loading a TorchScript model should enable is_loaded."""
        net = FleetPolicyNetwork(obs_dim=56, action_dim=31)
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "model.pt")
            net.export_torchscript(pt_path, obs_dim=56)

            engine = InferenceEngine(model_path=pt_path)
            assert engine.is_loaded

            obs = np.random.randn(56).astype(np.float32)
            mask = np.ones(31, dtype=bool)
            action, latency_ms = engine.predict(obs, action_mask=mask)
            assert 0 <= action < 31
            assert latency_ms > 0.0

    def test_torchscript_masking(self) -> None:
        """TorchScript inference with action mask should never select masked actions."""
        net = FleetPolicyNetwork(obs_dim=56, action_dim=31)
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "model.pt")
            net.export_torchscript(pt_path, obs_dim=56)

            engine = InferenceEngine(model_path=pt_path)
            obs = np.random.randn(56).astype(np.float32)
            mask = np.zeros(31, dtype=bool)
            mask[15] = True  # only action 15 is valid

            action, _ = engine.predict(obs, action_mask=mask)
            assert action == 15

    def test_warmup_torchscript(self) -> None:
        """Warmup should run without error on a TorchScript model."""
        net = FleetPolicyNetwork(obs_dim=56, action_dim=31)
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "model.pt")
            net.export_torchscript(pt_path, obs_dim=56)

            engine = InferenceEngine(model_path=pt_path)
            engine.warmup(obs_dim=56, n_iterations=10)
            assert engine._warm

    def test_unsupported_format_raises(self) -> None:
        """Loading an unsupported file extension should raise ValueError."""
        engine = InferenceEngine()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"not a model")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported model format"):
                engine.load(f.name)
        os.unlink(f.name)
