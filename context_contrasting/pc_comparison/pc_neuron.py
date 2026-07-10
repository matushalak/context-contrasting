from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def _as_vector(values: Any, *, length: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != length:
        raise ValueError(f"expected vector of length {length}, got shape={arr.shape}")
    return arr.copy()


def _relu(value: float) -> float:
    return float(max(0.0, value))


def _nonnegative(values: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(values, 0.0)


class _EMA:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.ema = 0.0

    def __call__(self, value: float) -> float:
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * float(value)
        return float(self.ema)

    def reset_state(self) -> None:
        self.ema = 0.0


class CorrectPCneuron:
    """PPE/NPE circuit that only reuses context-contrasting template parameters.

    Expected keys are the sampled paper/template values:
    ``w_ff``, ``w_fb``, ``w_pv``, ``w_lat``, ``lr_lat``, ``lr_fb``,
    ``pyc_decay`` and ``pv_decay``. There are no additional fitted gains,
    biases, adaptation terms, leaks, caps, or random per-cell parameters.
    """

    def __init__(self, cc_template_parameters: Mapping[str, Any]) -> None:
        self.w_LAT = float(_nonnegative(float(cc_template_parameters.get("w_lat", 0.0))))
        self.w_FB = _nonnegative(_as_vector(cc_template_parameters.get("w_fb", [0.0, 0.0, 0.0])))
        self.w_FF = _nonnegative(_as_vector(cc_template_parameters.get("w_ff", [0.0, 0.0, 0.0])))
        self.w_PV = _nonnegative(_as_vector(cc_template_parameters.get("w_pv", [0.0, 0.0, 0.0])))

        self.lr_fb = float(_nonnegative(float(cc_template_parameters.get("lr_fb", 0.0))))
        self.lr_lat = float(_nonnegative(float(cc_template_parameters.get("lr_lat", 0.0))))
        self.circuit = str(cc_template_parameters.get("circuit", "PPE"))
        self.use_lat_connection = True
        self.n_features = 3
        self.baseline_drive_sigma = float(_nonnegative(float(cc_template_parameters.get("baseline_drive_sigma", 0.0))))
        self.baseline_current_scale = 0.12
        self.rng = np.random.default_rng(int(cc_template_parameters.get("seed", 0)))
        self.baseline_current = float(
            self.rng.normal(0.0, self.baseline_current_scale * self.baseline_drive_sigma)
            if self.baseline_drive_sigma > 0.0
            else 0.0
        )

        pyc_decay = float(cc_template_parameters.get("pyc_decay", 0.05))
        pv_decay = float(cc_template_parameters.get("pv_decay", 0.5))
        self.pv = _EMA(pv_decay)
        self.pyramidal = _EMA(pyc_decay)

    def reset_state(self) -> None:
        self.pv.reset_state()
        self.pyramidal.reset_state()

    def _reset_state(self) -> None:
        self.reset_state()

    @property
    def w_ff(self) -> torch.Tensor:
        return torch.as_tensor(self.w_FF, dtype=torch.float32)

    @property
    def w_fb(self) -> torch.Tensor:
        return torch.as_tensor(self.w_FB, dtype=torch.float32)

    @property
    def w_lat(self) -> torch.Tensor:
        return torch.as_tensor([self.w_LAT], dtype=torch.float32)

    @property
    def w_pv_lat(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32)

    @property
    def W_pv(self) -> torch.Tensor:
        return torch.as_tensor(self.w_PV.reshape(1, -1), dtype=torch.float32)

    def PPE(self, x: np.ndarray, c: np.ndarray, *, update: bool = True) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        x = _as_vector(x)
        c = _as_vector(c)
        y_t = float(self.pyramidal.ema)
        pyc_drive = float(np.dot(self.w_FF, x))
        pv_response = self.pv(_relu(float(np.dot(self.w_PV, c))))
        inhibition = (self.w_LAT * pv_response) if self.use_lat_connection else 0.0
        v = pyc_drive - inhibition + self.baseline_current
        y_next = self.pyramidal(_relu(v))

        if update:
            self.w_LAT = float(_nonnegative(self.w_LAT + self.lr_lat * v * pv_response))
        return x, y_t, y_next, pv_response, c

    def NPE(self, x: np.ndarray, c: np.ndarray, *, update: bool = True) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        x = _as_vector(x)
        c = _as_vector(c)
        y_t = float(self.pyramidal.ema)
        pyc_drive = float(np.dot(self.w_FB, c))
        pv_response = self.pv(_relu(float(np.dot(self.w_PV, x))))
        inhibition = (self.w_LAT * pv_response) if self.use_lat_connection else 0.0
        v = pyc_drive - inhibition + self.baseline_current
        y_next = self.pyramidal(_relu(v))

        if update:
            self.w_FB = _nonnegative(self.w_FB - self.lr_fb * v * c)
        return x, y_t, y_next, pv_response, c

    def __call__(self, x: torch.Tensor, c: torch.Tensor, update:bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_np = np.asarray(x.detach().cpu(), dtype=float)
        c_np = np.asarray(c.detach().cpu(), dtype=float)
        if self.circuit == "NPE":
            _, y_t, y_next, pv_response, _ = self.NPE(x_np, c_np, update=update)
        else:
            _, y_t, y_next, pv_response, _ = self.PPE(x_np, c_np, update=update)
        return (
            torch.as_tensor(x_np, dtype=torch.float32),
            torch.as_tensor(y_t, dtype=torch.float32),
            torch.as_tensor(y_next, dtype=torch.float32),
            torch.as_tensor([pv_response], dtype=torch.float32),
            torch.as_tensor(c_np, dtype=torch.float32),
        )
