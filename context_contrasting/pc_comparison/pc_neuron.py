from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from context_contrasting.paper.neuron_utils import EMA, ThresholdReLU, nonnegative, randn_reparam


def _init_tensor(init: Mapping[str, Any] | torch.Tensor | list[float] | tuple[float, ...] | float, *, shape: tuple[int, ...]) -> torch.Tensor:
    if isinstance(init, Mapping):
        value = randn_reparam(size=(1,), **init)
    else:
        value = torch.as_tensor(init, dtype=torch.float32)
    value = value.to(dtype=torch.float32).reshape(shape)
    return nonnegative(value)


def _bounded_signed_delta(delta: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.where(delta >= 0.0, delta * (1.0 - weights), delta * weights)


class CorrectPCneuron(nn.Module):
    """Minimal PPE/NPE circuit using the same tensor-state contract as CCNeuron.

    The PC comparison keeps the circuit-specific effective inputs:
      - PPE: PyC gets sensory FF drive; PV gets context/prediction drive.
      - NPE: PyC gets context/prediction drive; PV gets sensory FF drive.

    Plasticity is only the balancing rule for the relevant PC synapse. The
    public attributes and `forward`/`update` contract intentionally match
    `paper.minimal_divisive.CCNeuron`, so repository helpers such as
    `run_experimental_phase`, `wide_to_long`, and `visualize_transition_panel`
    can be reused directly.

    NOTE:
    To have signed error, need to work with signed voltage, NOT rectified PyC output!!!
    If rectified output used in update equations, sign of the learning update cannot change and can only be one polarity OR zero.
    """

    def __init__(self, cc_template_parameters: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        params = dict(cc_template_parameters or {})
        params.update(kwargs)

        self.n_features = int(params.get("n_features", 3))
        self.n_context = int(params.get("n_context", self.n_features))
        self.n_pv = int(params.get("n_pv", 1))
        if self.n_features != 3 or self.n_context != 3 or self.n_pv != 1:
            raise ValueError("CorrectPCneuron currently expects 3 features/context channels and one PV cell.")

        seed = int(params.get("seed", 42))
        torch.manual_seed(seed)

        self.circuit = str(params.get("circuit", "PPE"))
        if self.circuit not in {"PPE", "NPE"}:
            raise ValueError("circuit must be 'PPE' or 'NPE'.")

        self.w_ff = _init_tensor(params.get("w_ff_init", {"mu": [0.0, 0.0, 0.0], "sigma": 0.0}), shape=(self.n_features,))
        self.w_fb = _init_tensor(params.get("w_fb_init", {"mu": [0.0, 0.0, 0.0], "sigma": 0.0}), shape=(self.n_context,))
        self.w_lat = _init_tensor(params.get("w_lat_init", {"mu": [0.0], "sigma": 0.0}), shape=(self.n_pv,))
        self.w_pv_lat = _init_tensor(params.get("w_pv_lat_init", {"mu": [0.0], "sigma": 0.0}), shape=(self.n_pv,))
        self.W_pv = _init_tensor(params.get("W_pv_init", {"mu": [0.0, 0.0, 0.0], "sigma": 0.0}), shape=(self.n_pv, self.n_features))

        receives_context = params.get("receives_context", (True, True, True))
        self.receives_context = torch.as_tensor(receives_context, dtype=torch.bool).reshape(self.n_context)
        self.w_fb = self.w_fb * self.receives_context

        self.lr_fb = float(nonnegative(torch.as_tensor(params.get("lr_fb", 0.0), dtype=torch.float32)))
        self.lr_ff = float(nonnegative(torch.as_tensor(params.get("lr_ff", 0.0), dtype=torch.float32)))
        self.lr_lat = float(nonnegative(torch.as_tensor(params.get("lr_lat", 0.0), dtype=torch.float32)))
        self.baseline_drive_mu = float(params.get("baseline_drive_mu", 0.0))
        self.baseline_drive_sigma = float(nonnegative(torch.as_tensor(params.get("baseline_drive_sigma", 0.0), dtype=torch.float32)))
        self.pv_noise_sigma = float(nonnegative(torch.as_tensor(params.get("pv_noise_sigma", 0.0), dtype=torch.float32)))

        pyc_decay = float(params.get("pyc_decay", 0.05))
        pv_decay = float(params.get("pv_decay", 0.5))
        self.activation = params.get(
            "activation",
            ThresholdReLU(threshold=0.0, subtractive=False, hasMax=True, maxValue=1.0),
        )
        self.pv = EMA(shape=(self.n_pv,), alpha=pv_decay)
        self.pyramidal = EMA(shape=(), alpha=pyc_decay)
        self.adapt = EMA(shape=(), alpha=pyc_decay * 0.2)

        self.use_FF_connection = bool(params.get("use_FF_connection", True))
        self.use_FB_connection = bool(params.get("use_FB_connection", True))
        self.use_lat_connection = bool(params.get("use_lat_connection", True))
        self.use_pv_connection = bool(params.get("use_pv_connection", True))
        self.use_pv_lat_connection = bool(params.get("use_pv_lat_connection", False))

        self._last_circuit = self.circuit
        self._last_voltage = torch.zeros((), dtype=torch.float32)
        self.pc_plasticity_mode = str(params.get("pc_plasticity_mode", "lat"))
        if self.pc_plasticity_mode not in {"lat", "ppe_ff_npe_fb"}:
            raise ValueError("pc_plasticity_mode must be 'lat' or 'ppe_ff_npe_fb'.")

    @torch.no_grad()
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.circuit == "PPE":
            return self.PPE(x, c, update=False)
        return self.NPE(x, c, update=False)

    @torch.no_grad()
    def PPE(self, x: torch.Tensor, c: torch.Tensor, *, update: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        In PPE circuit, PVs receive context and PyC's receive FF input
        '''
        x_t = torch.as_tensor(x, dtype=self.w_ff.dtype).reshape(self.n_features)
        c_t = torch.as_tensor(c, dtype=self.w_fb.dtype).reshape(self.n_context)
        y_t = self.pyramidal.ema
        a = self.adapt(self.pyramidal.ema)

        pv_fb = (self.W_pv @ c_t).reshape(-1) * self.use_pv_connection
        # NO pv_lat connection unlike in the CC model
        # pv_lat = y_t * self.w_pv_lat * self.use_pv_lat_connection
        # PV receive FB
        p = self.pv(
            self.activation(
                pv_fb
                # + pv_lat
                + randn_reparam(size=tuple(self.pv.ema.shape), mu=0.0, sigma=self.pv_noise_sigma)
            )
        )

        pyc_drive = torch.dot(self.w_ff, x_t) * self.use_FF_connection
        inhibition = torch.dot(self.w_lat, p) * self.use_lat_connection
        baseline_drive = randn_reparam(size=(), mu=self.baseline_drive_mu, sigma=self.baseline_drive_sigma)
        v = pyc_drive - inhibition + baseline_drive - a
        y_next = self.pyramidal(self.activation(v))

        self._last_circuit = "PPE"
        self._last_voltage = v.detach().clone()
        if update:
            if self.pc_plasticity_mode == "ppe_ff_npe_fb":
                self._update_ppe_ff(x_t)
            else:
                self._update_ppe(p)
        return x_t, y_t, y_next, p, c_t

    @torch.no_grad()
    def NPE(self, x: torch.Tensor, c: torch.Tensor, *, update: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ''''
        In NPE circuit, PVs receive FF input and PyC's receive context
        '''
        x_t = torch.as_tensor(x, dtype=self.w_ff.dtype).reshape(self.n_features)
        c_t = torch.as_tensor(c, dtype=self.w_fb.dtype).reshape(self.n_context)
        y_t = self.pyramidal.ema
        a = self.adapt(self.pyramidal.ema)

        pyc_drive = torch.dot(self.w_fb, c_t * self.receives_context) * self.use_FB_connection
        pv_ff = (self.W_pv @ x_t).reshape(-1) * self.use_pv_connection
        # NO pv_lat connection unlike in the CC model
        # pv_lat = y_t * self.w_pv_lat * self.use_pv_lat_connection
        p = self.pv(
            self.activation(
                pv_ff
                # + pv_lat
                + randn_reparam(size=tuple(self.pv.ema.shape), mu=0.0, sigma=self.pv_noise_sigma)
            )
        )

        inhibition = torch.dot(self.w_lat, p) * self.use_lat_connection
        baseline_drive = randn_reparam(size=(), mu=self.baseline_drive_mu, sigma=self.baseline_drive_sigma)
        v = pyc_drive - inhibition
        y_next = self.pyramidal(self.activation(v + baseline_drive - a))

        self._last_circuit = "NPE"
        self._last_voltage = v.detach().clone()
        if update:
            # self._update_npe(p)
            self._update_npe(c_t)
        return x_t, y_t, y_next, p, c_t

    @torch.no_grad()
    def update(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        y_next: torch.Tensor,
        pv_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> None:
        if self._last_circuit == "PPE":
            if self.pc_plasticity_mode == "ppe_ff_npe_fb":
                self._update_ppe_ff(x_t)
            else:
                self._update_ppe(pv_t)
        else:
            # self._update_npe(pv_t)
            self._update_npe(c_t)

    @torch.no_grad()
    def _update_ppe(self, pv_t: torch.Tensor) -> None:
        if self.pc_plasticity_mode == "ppe_ff_npe_fb":
            return
        raw_delta = self.lr_lat * self._last_voltage * pv_t.reshape(self.n_pv)
        self.w_lat = nonnegative(self.w_lat + _bounded_signed_delta(raw_delta, self.w_lat))

    @torch.no_grad()
    def _update_ppe_ff(self, x_t: torch.Tensor) -> None:
        raw_delta = -self.lr_ff * self._last_voltage * x_t.reshape(self.n_features)
        self.w_ff = nonnegative(self.w_ff + _bounded_signed_delta(raw_delta, self.w_ff))


    # @torch.no_grad()
    # def _update_npe(self, pv_t: torch.Tensor) -> None:
    #     raw_delta = self.lr_lat * self._last_voltage * pv_t.reshape(self.n_pv)
    #     self.w_lat = nonnegative(self.w_lat + _bounded_signed_delta(raw_delta, self.w_lat))

    @torch.no_grad()
    def _update_npe(self, c_t: torch.Tensor) -> None:
        raw_delta = -self.lr_fb * self._last_voltage * c_t.reshape(self.n_context) * self.receives_context
        self.w_fb = nonnegative(self.w_fb + _bounded_signed_delta(raw_delta, self.w_fb)) * self.receives_context

    def reset_state(self) -> None:
        self.pv.reset_state()
        self.pyramidal.reset_state()
        self.adapt.reset_state()

    def _reset_state(self) -> None:
        self.reset_state()
