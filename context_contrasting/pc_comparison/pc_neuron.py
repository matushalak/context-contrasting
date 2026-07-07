from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from context_contrasting.mini_network.utils import EMA


CircuitType = Literal["PPE", "NPE"]


@dataclass(frozen=True)
class PCResponse:
    response: float
    pyc_drive: float
    pv_response: float
    inhibition: float
    residual: float


def _as_vector(values: Any, *, length: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != length:
        raise ValueError(f"expected vector of length {length}, got shape={arr.shape}")
    return arr.copy()


def _relu(value: float) -> float:
    return float(max(0.0, value))


class CorrectPCneuron:
    """"
    Reusing parameters from context-contrasting circuit population
    """
    def __init__(self, CC_template_parameters: Mapping[str, Any]) -> None:
        self.w_LAT = float(CC_template_parameters.get("w_lat", 0.0))
        self.w_FB = _as_vector(CC_template_parameters.get("w_fb", [0.0, 0.0, 0.0]))
        self.w_FF = _as_vector(CC_template_parameters.get("w_ff", [0.0, 0.0, 0.0]))
        self.activation = CC_template_parameters.get("activation", lambda x: max(0.0, x))

        self.pv = EMA(...)
        self.pyramidal = EMA(...)
        self.adapt = EMA(...)
    

        self.lr_fb = nonnegative(float(CC_template_parameters.get("lr_fb", 0.0)))
        self.lr_ff = nonnegative(float(CC_template_parameters.get("lr_ff", 0.0)) )

    def PPE(self, x, c):
        y_t = self.pyramidal.ema
        a = self.adapt(self.pyramidal.ema)
        pyc_drive = float(np.dot(self.w_FF, x))
        pv_response = self.pv(self.activation(float(np.dot(self.w_FB, c)) + baseline))
        inhibition = self.w_LAT * pv_response
        y_next = self.pyramidal(self.activation(pyc_drive - inhibition + baseline - a))
        
        # already contains update
        self.w_LAT += self.lr_ff * y_t * pv_response # dw = (ff-fb) * fb
        return x, y_t, y_next, pv_response, c
    
    def NPE(self, x, c):
        y_t = self.pyramidal.ema
        a = self.adapt(self.pyramidal.ema)
        pyc_drive = float(np.dot(self.w_FB, c))
        pv_response = self.pv(self.activation(float(np.dot(self.w_FF, x)) + baseline))
        inhibition = self.w_LAT * pv_response
        y_next = self.pyramidal(self.activation(pyc_drive - inhibition + baseline - a))

        # already contains update
        self.w_FB += self.lr_fb * y_t * c # dw = (fb-ff) * fb
        return x, y_t, y_next, pv_response, c


# TOO TWEAKY and bloated
class PCNeuron:
    """Static subtractive PPE/NPE rate unit for the model-comparison panels.

    The class deliberately uses only the synapses that are allowed for each
    canonical circuit:

    * PPE: PyC gets feedforward drive via ``w_ff``; a context-driven PV cell uses
      ``w_pv_fb`` and inhibits PyC through plastic ``w_lat``.
    * NPE: PyC gets context drive via plastic ``w_fb``; a feedforward-driven PV
      cell uses ``w_pv_ff`` and inhibits PyC through fixed ``w_lat``.

    ``W_pv`` and ``w_pv_lat`` from the context-contrasting model are intentionally
    not part of the circuit unless the caller explicitly maps ``W_pv`` into
    ``w_pv_ff`` before constructing an NPE unit.
    """

    def __init__(
        self,
        neuron_id: int,
        neuron_type: CircuitType,
        parameters: Mapping[str, Any],
    ) -> None:
        if neuron_type not in {"PPE", "NPE"}:
            raise ValueError("neuron_type must be 'PPE' or 'NPE'.")
        self.neuron_id = int(neuron_id)
        self.neuron_type: CircuitType = neuron_type
        self.w_ff = _as_vector(parameters.get("w_ff", [0.0, 0.0, 0.0]))
        self.w_fb = _as_vector(parameters.get("w_fb", [0.0, 0.0, 0.0]))
        self.w_pv_fb = _as_vector(parameters.get("w_pv_fb", self.w_fb))
        self.w_pv_ff = _as_vector(parameters.get("w_pv_ff", self.w_ff))
        self.w_lat = float(np.asarray(parameters.get("w_lat", 0.0), dtype=float).reshape(-1)[0])
        self.bias = float(parameters.get("bias", 0.0))
        self.pv_bias = float(parameters.get("pv_bias", 0.0))
        self.ff_gain = float(parameters.get("ff_gain", 1.0))
        self.fb_gain = float(parameters.get("fb_gain", 1.0))
        self.pv_gain = float(parameters.get("pv_gain", 1.0))
        self.lat_gain = float(parameters.get("lat_gain", 1.0))
        self.lr_lat = float(parameters.get("lr_lat", 0.0))
        self.lr_fb = float(parameters.get("lr_fb", 0.0))
        self.max_weight = float(parameters.get("max_weight", 5.0))
        response_max = parameters.get("response_max")
        self.response_max = None if response_max is None else float(response_max)

    def activate(self, ff_input: np.ndarray, context_input: np.ndarray) -> float:
        return self.components(ff_input, context_input).response

    def components(self, ff_input: np.ndarray, context_input: np.ndarray) -> PCResponse:
        ff_input = _as_vector(ff_input)
        context_input = _as_vector(context_input)

        if self.neuron_type == "PPE":
            pyc_drive = self.ff_gain * float(np.dot(self.w_ff, ff_input))
            pv_response = self.pv_gain * _relu(float(np.dot(self.w_pv_fb, context_input)) - self.pv_bias)
        else:
            pyc_drive = self.fb_gain * float(np.dot(self.w_fb, context_input))
            pv_response = self.pv_gain * _relu(float(np.dot(self.w_pv_ff, ff_input)) - self.pv_bias)

        inhibition = self.lat_gain * self.w_lat * pv_response
        residual = pyc_drive - inhibition
        response = _relu(residual - self.bias)
        if self.response_max is not None:
            response = min(response, self.response_max)
        return PCResponse(
            response=response,
            pyc_drive=pyc_drive,
            pv_response=pv_response,
            inhibition=inhibition,
            residual=residual,
        )

    def train(
        self,
        ff_input: np.ndarray,
        context_input: np.ndarray,
        *,
        plasticity_context: np.ndarray | None = None,
        repetitions: int = 1,
    ) -> None:
        if repetitions < 1:
            return
        ff_input = _as_vector(ff_input)
        context_input = _as_vector(context_input)
        plasticity_context = _as_vector(plasticity_context) if plasticity_context is not None else context_input
        for _ in range(int(repetitions)):
            parts = self.components(ff_input, context_input)
            if self.neuron_type == "PPE":
                # Balance expected sensory evidence with context-driven inhibition.
                delta = self.lr_lat * parts.residual * parts.pv_response
                self.w_lat = float(np.clip(self.w_lat + delta, 0.0, self.max_weight))
            else:
                # Strengthen the active generative/context channel when the NPE
                # residual remains positive after sensory-driven inhibition.
                delta = self.lr_fb * parts.residual * plasticity_context
                self.w_fb = np.clip(self.w_fb + delta, 0.0, self.max_weight)

    def update_parameters(self, new_parameters: Mapping[str, Any]) -> None:
        for key, value in new_parameters.items():
            if key in {"w_ff", "w_fb", "w_pv_fb", "w_pv_ff"}:
                setattr(self, key, _as_vector(value))
            elif hasattr(self, key):
                setattr(self, key, float(value))
            else:
                raise KeyError(f"unknown PCNeuron parameter: {key}")

    def get_info(self) -> dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "neuron_type": self.neuron_type,
            "w_ff": self.w_ff.tolist(),
            "w_fb": self.w_fb.tolist(),
            "w_pv_fb": self.w_pv_fb.tolist(),
            "w_pv_ff": self.w_pv_ff.tolist(),
            "w_lat": self.w_lat,
            "bias": self.bias,
            "pv_bias": self.pv_bias,
            "ff_gain": self.ff_gain,
            "fb_gain": self.fb_gain,
            "pv_gain": self.pv_gain,
            "lat_gain": self.lat_gain,
            "lr_lat": self.lr_lat,
            "lr_fb": self.lr_fb,
            "max_weight": self.max_weight,
            "response_max": self.response_max,
        }
