"""Seed/neuronal uncertainty and ensemble aggregation contracts."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_v2.plot import (SECTORS, SOURCES, LABELS, average_traces, cosine_similarities,
                          model_transition_vectors, plot_vectors, summarize_traces,
                          trace_observations)
from paper_v2.simulate import load_config, seeded_cell, simulate, simulate_ensemble


def examples():
    model, data = [], []
    for source in SOURCES:
        for i, sector in enumerate(SECTORS):
            for neuron, vector in enumerate(((1., 0.), (0., 1.))):
                data.append(dict(record_type="transition", source=source, sector=sector,
                                 observation_id=10 * i + neuron, delta_NO=vector[0], delta_O=vector[1]))
            for seed, vector in enumerate(((1., 0.), (0., 3.))):
                for image, offset in ((1, -5.), (2, 5.)):
                    for condition in LABELS:
                        for j, response in enumerate(("NO", "O")):
                            value = 0. if condition == "naive" else vector[j]
                            model.append(dict(record_type="trace", source=source, sector=sector,
                                              observation_id=i, seed=seed, image_id=image,
                                              condition_key=condition, response_type=response,
                                              time_seconds=0., response=value + offset))
    return pd.DataFrame(data), pd.DataFrame(model)


class EnsembleTests(unittest.TestCase):
    def test_images_are_averaged_before_seed_uncertainty(self):
        _, model = examples()
        observations = trace_observations(model)
        keys = ["source", "sector", "condition_key", "response_type", "time_seconds"]
        summary = summarize_traces(observations, keys)
        r = summary.query('source=="familiar" and sector=="+O axis" and condition_key=="expert" and response_type=="O"').iloc[0]
        self.assertEqual(r.n, 2)
        self.assertEqual(r.response, 1.5)
        self.assertAlmostEqual(r.sd, 3 / np.sqrt(2))
        self.assertAlmostEqual(r["sem"], 1.5)

    def test_neurons_not_images_are_data_units(self):
        rows = [dict(record_type="trace", source="familiar", sector="+O axis",
                     condition_key="expert", response_type="NO", time_seconds=0.,
                     observation_id=i, image_id=j, response=x)
                for i, values in enumerate(((-10., 10.), (2.,))) for j, x in enumerate(values)]
        r = average_traces(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(r.n, 2)
        self.assertEqual(r.response, 1.)
        self.assertAlmostEqual(r.sd, np.sqrt(2))
        self.assertAlmostEqual(r["sem"], 1.)

    def test_weighted_uncertainty_combines_within_seed_first(self):
        _, model = examples()
        weights = pd.Series(1 / 3, index=pd.MultiIndex.from_product([SOURCES, SECTORS], names=["source", "sector"]))
        r = average_traces(model, weights).query('source=="familiar" and condition_key=="expert" and response_type=="O"').iloc[0]
        self.assertEqual(r.n, 2)
        self.assertAlmostEqual(r.response, 1.5)
        # All sectors co-vary across seeds: pooling them would give wrong SD/SEM.
        self.assertAlmostEqual(r.sd, 3 / np.sqrt(2))
        self.assertAlmostEqual(r["sem"], 1.5)

    def test_incomplete_weighted_replicate_is_not_silently_accepted(self):
        _, model = examples()
        model = model[~(model.seed.eq(1) & model.sector.eq("+O axis"))]
        weights = pd.Series(1 / 3, index=pd.MultiIndex.from_product([SOURCES, SECTORS], names=["source", "sector"]))
        with self.assertRaisesRegex(ValueError, "every sector"):
            average_traces(model, weights)

    def test_seed_vectors_are_paired_before_averaging(self):
        _, model = examples()
        vectors = model_transition_vectors(model, "expert")
        self.assertEqual(len(vectors), 12)
        r = vectors.query('source=="familiar" and sector=="+O axis"').sort_values("seed")
        np.testing.assert_allclose(r[["delta_NO", "delta_O"]], [[1, 0], [0, 3]])

    def test_cosine_uses_mean_model_against_each_neuron(self):
        data, model = examples()
        values = cosine_similarities(data, model)
        values = values.query('source=="familiar" and condition=="Expert" and sector=="+O"')
        self.assertEqual(len(values), 2)
        np.testing.assert_allclose(values.similarity, np.array([1, 3]) / np.sqrt(10))

    def test_model_ellipses_and_sem_scaling(self):
        data, model = examples()
        widths = []
        for uncertainty in ("sd", "sem"):
            with patch("paper_v2.plot.save") as save:
                plot_vectors(data, model, "unused", uncertainty)
                fig = save.call_args.args[0]
                patches = fig.axes[1].patches  # familiar model panel
                self.assertEqual(len(patches), 3)
                widths.append(patches[0].width)
                plt.close(fig)
        self.assertAlmostEqual(widths[0] / widths[1], np.sqrt(2))

    def test_legacy_single_run_has_no_fabricated_uncertainty(self):
        _, model = examples()
        model = model[model.seed.eq(0)].drop(columns="seed")
        weights = pd.Series(1 / 3, index=pd.MultiIndex.from_product([SOURCES, SECTORS], names=["source", "sector"]))
        summary = average_traces(model, weights)
        self.assertTrue(summary.n.eq(1).all())
        self.assertTrue(summary.sd.isna().all())
        self.assertEqual(len(model_transition_vectors(model, "expert")), 6)

    def test_seed_override_does_not_mutate_configs_and_is_reproducible(self):
        cells, protocol = load_config()
        original_seed = cells[0]["parameters"]["seed"]
        cell = seeded_cell(cells[0], 123)
        self.assertEqual(cells[0]["parameters"]["seed"], original_seed)
        protocol.update(steps_per_trial=40, steps_per_second=10, training_trials=1,
                        test_trials=2, rng_reference_trials=1)
        a, b = simulate(cell, protocol), simulate(cell, protocol)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a.seed.eq(123).all())
        v = model_transition_vectors(a, "expert").iloc[0]
        saved = a[a.record_type.eq("transition")].iloc[0]
        np.testing.assert_allclose(v[["delta_NO", "delta_O"]].astype(float),
                                   saved[["delta_NO", "delta_O"]].astype(float))

    def test_serial_and_parallel_ensembles_are_identical(self):
        cells, protocol = load_config()
        protocol.update(steps_per_trial=40, steps_per_second=10, training_trials=1,
                        test_trials=2, rng_reference_trials=1)
        serial = simulate_ensemble(cells[:1], protocol, seeds=[11, 12], jobs=1)
        parallel = simulate_ensemble(cells[:1], protocol, seeds=[11, 12], jobs=2)
        pd.testing.assert_frame_equal(serial, parallel)
        self.assertEqual(set(parallel.seed), {11, 12})


if __name__ == "__main__":
    unittest.main()
