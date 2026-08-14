from __future__ import annotations

import unittest

import numpy as np
import torch

from context_contrasting.paper import model_scatter
from context_contrasting.pc_comparison.pc_convergence import convergence_summary
from context_contrasting.pc_comparison.pc_neuron import CorrectPCneuron
from context_contrasting.pc_comparison.pc_templates import (
    DEFAULT_BASELINE_DRIVE,
    DEFAULT_BASELINE_DRIVE_SIGMA,
    DEFAULT_CONVERGENCE_TOLERANCE,
    FULL_PROTOCOL_CALIBRATED_LEARNING_RATE,
    sample_shared_pc_configs,
    scaled_learning_rate,
)
from context_contrasting.pc_comparison.run_pc_comparison import _model_params_from_row


class PCComparisonTests(unittest.TestCase):
    def test_ppe_and_npe_are_exact_input_swaps(self) -> None:
        params = {
            "pyc_excitatory_init": [0.7, 0.2, 0.4],
            "pv_excitatory_init": [0.3, 0.8, 0.5],
            "w_lat_init": [0.6],
            "learning_rate": 0.1,
            "pyc_decay": 0.05,
            "pv_decay": 0.5,
        }
        ppe = CorrectPCneuron(circuit="PPE", **params)
        npe = CorrectPCneuron(circuit="NPE", **params)
        x = torch.tensor([1.0, 0.2, 0.0])
        c = torch.tensor([0.1, 0.7, 0.4])

        ppe(x, c)
        npe(c, x)

        self.assertAlmostEqual(ppe.signed_prediction_error, npe.signed_prediction_error, places=7)
        np.testing.assert_allclose(ppe.w_ff.numpy(), npe.w_fb.numpy(), atol=0.0)
        np.testing.assert_allclose(ppe.W_pv.numpy(), npe.W_pv.numpy(), atol=0.0)
        np.testing.assert_allclose(ppe.w_lat.numpy(), npe.w_lat.numpy(), atol=0.0)

    def test_matched_training_produces_identical_plastic_weights(self) -> None:
        configs = sample_shared_pc_configs(
            n_samples=300,
            seed=7151,
            n_steps_per_phase=400,
            learning_rate=FULL_PROTOCOL_CALIBRATED_LEARNING_RATE,
        )
        # This is the limiting row in the full-population calibration.
        row = configs.iloc[36]
        ppe = CorrectPCneuron(_model_params_from_row(row, circuit="PPE"))
        npe = CorrectPCneuron(_model_params_from_row(row, circuit="NPE"))
        x_train, c_train = model_scatter._build_model_scatter_training_stimuli(
            n_steps_per_phase=400,
            n_trials=7,
            order="randomized",
            seed=7151,
        )
        for x_t, c_t in zip(x_train, c_train, strict=True):
            ppe_state = ppe(x_t, c_t)
            npe_state = npe(x_t, c_t)
            ppe.update(*ppe_state)
            npe.update(*npe_state)

        np.testing.assert_allclose(ppe.w_ff.numpy(), npe.w_fb.numpy(), atol=1e-7)
        steady_target = ppe.w_lat.item() * ppe.W_pv.numpy().reshape(-1)
        self.assertLessEqual(float(np.max(np.abs(ppe.w_ff.numpy()[:2] - steady_target[:2]))), 0.005)

    def test_full_protocol_calibrated_rate_converges_population(self) -> None:
        configs = sample_shared_pc_configs(
            n_samples=300,
            seed=7151,
            n_steps_per_phase=400,
            learning_rate=0.0,
        )
        full = convergence_summary(
            configs,
            reference_learning_rate=FULL_PROTOCOL_CALIBRATED_LEARNING_RATE,
            n_steps_per_phase=400,
            training_trials=7,
            seed=7151,
            tolerance=DEFAULT_CONVERGENCE_TOLERANCE,
        )
        self.assertTrue(full.converged)
        self.assertEqual(full.converged_fraction, 1.0)

        quick_candidate_at_full = convergence_summary(
            configs,
            reference_learning_rate=0.27477161843436204,
            n_steps_per_phase=400,
            training_trials=7,
            seed=7151,
            tolerance=DEFAULT_CONVERGENCE_TOLERANCE,
        )
        self.assertFalse(quick_candidate_at_full.converged)

    def test_learning_rate_scaling_uses_400_step_reference(self) -> None:
        self.assertAlmostEqual(scaled_learning_rate(0.25, 100), 1.0)
        self.assertAlmostEqual(scaled_learning_rate(0.25, 400), 0.25)

    def test_shared_baseline_is_passed_to_both_circuits(self) -> None:
        configs = sample_shared_pc_configs(
            n_samples=1,
            seed=7151,
            n_steps_per_phase=400,
        )
        row = configs.iloc[0]
        self.assertEqual(float(row["baseline_drive_mu"]), DEFAULT_BASELINE_DRIVE)
        self.assertEqual(float(row["baseline_drive_sigma"]), DEFAULT_BASELINE_DRIVE_SIGMA)
        for circuit in ("PPE", "NPE"):
            model = CorrectPCneuron(_model_params_from_row(row, circuit=circuit))
            self.assertEqual(model.baseline_drive_mu, DEFAULT_BASELINE_DRIVE)
            self.assertAlmostEqual(model.baseline_drive_sigma, DEFAULT_BASELINE_DRIVE_SIGMA)

    def test_response_noise_can_be_replayed(self) -> None:
        model = CorrectPCneuron(
            circuit="PPE",
            pyc_excitatory_init=[0.7, 0.2, 0.4],
            pv_excitatory_init=[0.3, 0.8, 0.5],
            w_lat_init=[0.6],
            baseline_drive_mu=DEFAULT_BASELINE_DRIVE,
            baseline_drive_sigma=DEFAULT_BASELINE_DRIVE_SIGMA,
            seed=7151,
        )
        x = torch.tensor([1.0, 0.0, 0.0])
        c = torch.tensor([1.0, 0.0, 0.0])
        initial_noise_state = model.get_noise_state()
        first = [float(model(x, c)[2].item()) for _ in range(20)]
        model.reset_state()
        model.set_noise_state(initial_noise_state)
        replay = [float(model(x, c)[2].item()) for _ in range(20)]
        np.testing.assert_allclose(first, replay, atol=0.0)


if __name__ == "__main__":
    unittest.main()
