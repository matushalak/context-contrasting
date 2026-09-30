"""Scientific parity checks for the standalone mean-field equations."""

import unittest

import torch

from thesis.population.minimal_divisive import CCNeuron as ReferenceNeuron
from paper_v2.model import CCNeuron
from paper_v2.simulate import load_config


class MeanFieldTests(unittest.TestCase):
    def test_six_cells_match_reference_dynamics_and_learning(self):
        for cell in load_config()[0]:
            with self.subTest(cell=cell["id"]):
                p = cell["parameters"]
                weights = ("w_ff", "w_fb", "w_lat", "w_pv_lat", "W_pv")
                config = {k: v for k, v in p.items() if k not in weights}
                config.update({f"{k}_init": dict(mu=p[k], sigma=0.) for k in weights})
                config.update(apical_drive_subtractive=True, use_ff_activity_accumulator=True,
                              pv_plasticity=False, pv_lat_plasticity=False,
                              FBrule="undampened-anti-Hebbian")
                reference = ReferenceNeuron(**config)
                reference.fb_specificity = torch.eye(3)
                rng = torch.random.get_rng_state()
                model = CCNeuron(p)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                for step in range(400):
                    x = torch.eye(3)[step % 3] if step % 100 >= 50 else torch.zeros(3)
                    c = (x * .4 + .6) / 2.2 if x.sum() else torch.zeros(3)
                    rng = torch.random.get_rng_state()
                    state = reference(x, c)
                    reference.update(*state)
                    torch.random.set_rng_state(rng)
                    self.assertEqual(model.step(x, c, learn=True), state[2].item())
                    for name in weights:
                        torch.testing.assert_close(getattr(model, name), getattr(reference, name), rtol=0, atol=0)
                accumulator = model.accumulator.clone()
                model.reset()
                self.assertEqual(model.y.item(), 0)
                torch.testing.assert_close(model.accumulator, accumulator, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
