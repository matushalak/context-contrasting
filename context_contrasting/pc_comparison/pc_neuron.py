from __future__ import annotations

from typing import Any, Literal, Mapping

import torch
import torch.nn as nn

from context_contrasting.paper.neuron_utils import EMA, ThresholdReLU, nonnegative, randn_reparam


Circuit = Literal["PPE", "NPE"]


def _init_tensor(
    init: Mapping[str, Any] | torch.Tensor | list[float] | tuple[float, ...] | float,
    *,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if isinstance(init, Mapping):
        value = randn_reparam(size=(1,), **init)
    else:
        value = torch.as_tensor(init, dtype=torch.float32)
    return nonnegative(value.to(dtype=torch.float32).reshape(shape))


def _bounded_signed_delta(delta: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.where(delta >= 0.0, delta * (1.0 - weights), delta * weights)


class CorrectPCneuron(nn.Module):
    """Matched PPE/NPE error neuron with one shared parameter contract.

    ``pyc_excitatory_init`` is sensory ``w_FF`` in PPE and contextual ``w_FB``
    in NPE. ``pv_excitatory_init`` is contextual PV excitation in PPE and
    sensory PV excitation in NPE. The only plastic synapse is the direct PyC
    excitation, updated by the same anti-Hebbian rule in both circuits.

    The public ``w_ff``, ``w_fb``, ``w_lat``, ``w_pv_lat`` and ``W_pv``
    attributes preserve the state contract used by the paper's simulation and
    plotting helpers.
    """

    def __init__(self, parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        params = dict(parameters or {})
        params.update(kwargs)

        self.n_features = int(params.get("n_features", 3))
        self.n_context = int(params.get("n_context", self.n_features))
        self.n_pv = int(params.get("n_pv", 1))
        if self.n_features != self.n_context or self.n_pv != 1:
            raise ValueError("CorrectPCneuron expects equal feature/context dimensions and one PV cell.")

        self.circuit: Circuit = str(params.get("circuit", "PPE"))  # type: ignore[assignment]
        if self.circuit not in {"PPE", "NPE"}:
            raise ValueError("circuit must be 'PPE' or 'NPE'.")

        seed = int(params.get("seed", 42))
        torch.manual_seed(seed)
        self._noise_generator = torch.Generator(device="cpu")
        self._noise_generator.manual_seed(seed)

        pyc_excitatory = _init_tensor(
            params.get("pyc_excitatory_init", [0.0] * self.n_features),
            shape=(self.n_features,),
        )
        self.w_ff = pyc_excitatory.clone() if self.circuit == "PPE" else torch.zeros_like(pyc_excitatory)
        self.w_fb = pyc_excitatory.clone() if self.circuit == "NPE" else torch.zeros_like(pyc_excitatory)
        self.W_pv = _init_tensor(
            params.get("pv_excitatory_init", [0.0] * self.n_features),
            shape=(self.n_pv, self.n_features),
        )
        self.w_lat = _init_tensor(params.get("w_lat_init", [0.0]), shape=(self.n_pv,))
        self.w_pv_lat = torch.zeros(self.n_pv, dtype=torch.float32)
        self.receives_context = torch.ones(self.n_context, dtype=torch.bool)

        self.learning_rate = float(
            nonnegative(torch.as_tensor(params.get("learning_rate", 0.0), dtype=torch.float32))
        )

        self.baseline_drive_mu = float(params.get("baseline_drive_mu", 0.0))
        self.baseline_drive_sigma = float(
            nonnegative(torch.as_tensor(params.get("baseline_drive_sigma", 0.0), dtype=torch.float32))
        )
        self.pv_noise_sigma = float(
            nonnegative(torch.as_tensor(params.get("pv_noise_sigma", 0.0), dtype=torch.float32))
        )

        pyc_decay = float(params.get("pyc_decay", 0.05))
        pv_decay = float(params.get("pv_decay", 0.5))
        self.activation = params.get(
            "activation",
            ThresholdReLU(threshold=0.0, subtractive=False, hasMax=True, maxValue=1.0),
        )
        self.pv = EMA(shape=(self.n_pv,), alpha=pv_decay)
        self.pyramidal = EMA(shape=(), alpha=pyc_decay)
        self.adapt = EMA(shape=(), alpha=pyc_decay * 0.2)

        self.use_FF_connection = True
        self.use_FB_connection = True
        self.use_lat_connection = True
        self.use_pv_connection = True
        self.use_pv_lat_connection = False

        self._last_signed_error = torch.zeros((), dtype=torch.float32)
        self._last_presynaptic = torch.zeros(self.n_features, dtype=torch.float32)

    @property
    def signed_prediction_error(self) -> float:
        return float(self._last_signed_error.item())

    def get_noise_state(self) -> torch.Tensor:
        return self._noise_generator.get_state().clone()

    def set_noise_state(self, state: torch.Tensor) -> None:
        self._noise_generator.set_state(state.clone())

    def _direct_weights(self) -> torch.Tensor:
        return self.w_ff if self.circuit == "PPE" else self.w_fb

    def _set_direct_weights(self, weights: torch.Tensor) -> None:
        if self.circuit == "PPE":
            self.w_ff = weights
        else:
            self.w_fb = weights

    def _sample_noise(self, shape: tuple[int, ...], *, mu: float, sigma: float) -> torch.Tensor:
        if sigma == 0.0:
            return torch.full(shape, mu, dtype=self.w_ff.dtype)
        return mu + sigma * torch.randn(
            shape,
            dtype=self.w_ff.dtype,
            generator=self._noise_generator,
        )

    @torch.no_grad()
    def _circuit_step(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        update: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = torch.as_tensor(x, dtype=self.w_ff.dtype).reshape(self.n_features)
        c_t = torch.as_tensor(c, dtype=self.w_fb.dtype).reshape(self.n_context)
        direct_input, pv_input = (x_t, c_t) if self.circuit == "PPE" else (c_t, x_t)
        y_t = self.pyramidal.ema
        a = self.adapt(self.pyramidal.ema)

        pv_drive = (self.W_pv @ pv_input).reshape(-1) * self.use_pv_connection
        p = self.pv(
            self.activation(
                pv_drive
                + self._sample_noise(
                    tuple(self.pv.ema.shape),
                    mu=0.0,
                    sigma=self.pv_noise_sigma,
                )
            )
        )
        pyc_drive = torch.dot(self._direct_weights(), direct_input)
        inhibition = torch.dot(self.w_lat, p) * self.use_lat_connection
        signed_error = pyc_drive - inhibition
        baseline_drive = self._sample_noise(
            (),
            mu=self.baseline_drive_mu,
            sigma=self.baseline_drive_sigma,
        )
        y_next = self.pyramidal(self.activation(signed_error + baseline_drive - a))

        self._last_signed_error = signed_error.detach().clone()
        self._last_presynaptic = direct_input.detach().clone()
        if update:
            self._update_direct_excitation()
        return x_t, y_t, y_next, p, c_t

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._circuit_step(x, c, update=False)

    @torch.no_grad()
    def PPE(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        update: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.circuit != "PPE":
            raise RuntimeError("PPE() can only be called on a PPE model.")
        return self._circuit_step(x, c, update=update)

    @torch.no_grad()
    def NPE(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        update: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.circuit != "NPE":
            raise RuntimeError("NPE() can only be called on an NPE model.")
        return self._circuit_step(x, c, update=update)

    @torch.no_grad()
    def update(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        y_next: torch.Tensor,
        pv_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> None:
        del x_t, y_t, y_next, pv_t, c_t
        self._update_direct_excitation()

    @torch.no_grad()
    def _update_direct_excitation(self) -> None:
        weights = self._direct_weights()
        raw_delta = -self.learning_rate * self._last_signed_error * self._last_presynaptic
        updated = nonnegative(weights + _bounded_signed_delta(raw_delta, weights))
        self._set_direct_weights(updated)

    def reset_state(self) -> None:
        self.pv.reset_state()
        self.pyramidal.reset_state()
        self.adapt.reset_state()
        self._last_signed_error.zero_()
        self._last_presynaptic.zero_()

    def _reset_state(self) -> None:
        self.reset_state()
