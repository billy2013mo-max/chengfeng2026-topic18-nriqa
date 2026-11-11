#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/decoder55.py
========================
A small, auditable decision layer that maps a (N, 6, 4) simplex to labels and is fitted
*directly* on the competition objective by coordinate ascent:

  1. per-dimension log-bias  b[d, k]   (k = P2/P1/P0; 无问题 bias fixed at 0)
         pred[d] = argmax_k  log p[d, k] + b[d, k]
  2. global clean gate  t_clean:  if  prod_d p[d, 无问题] >= t_clean  ->  all six = 无问题
     (directly targets Doc2_Strict Clean2, which needs *all six* dims right on clean images)
  3. P0 recall floor  f0[d]:  if p[d, P0] >= f0[d] -> P0, regardless of argmax
     (S_p0 weights recall by 0.5-0.6, so a lower-than-argmax threshold is usually optimal)

`fit()` never sees the rows it is evaluated on when called through `nested_oof_decode()`.
17-23 free parameters total, so the over-fitting surface is tiny compared to 16-dim threshold
searches that were rejected earlier (memory: 50.27 vs 50.48).
"""
from typing import Dict, Optional, Sequence

import numpy as np

from common55 import FULL_DIMS, P0_ONLY_DIMS, renorm, score

BIAS_GRID = np.round(np.arange(-2.0, 2.01, 0.25), 3)
GATE_GRID = np.array([0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.01])
FLOOR_GRID = np.array([1.01, 0.60, 0.50, 0.42, 0.35, 0.30, 0.25, 0.20, 0.15])


class Decoder55:
    def __init__(self, objective: str = "doc1_final"):
        assert objective in ("blend", "doc1_total", "doc2_strict_total", "doc1_final")
        self.objective = objective
        self.bias = np.zeros((6, 4))
        self.t_clean = 0.0
        self.floor = np.full(6, 1.01)

    # ------------------------------------------------------------------
    def predict(self, s: np.ndarray) -> np.ndarray:
        s = renorm(s)
        lg = np.log(s) + self.bias[None]
        lg[:, P0_ONLY_DIMS, 1] = -1e9
        lg[:, P0_ONLY_DIMS, 2] = -1e9
        pred = lg.argmax(-1)
        for d in range(6):
            pred[s[:, d, 3] >= self.floor[d], d] = 3
        if self.t_clean > 0:
            q_clean = np.prod(s[:, :, 0], axis=1)
            pred[q_clean >= self.t_clean] = 0
        return pred

    def _obj(self, s, gold) -> float:
        return float(score(self.predict(s), gold)[self.objective])

    # ------------------------------------------------------------------
    def fit(self, s: np.ndarray, gold: np.ndarray, sweeps: int = 3, verbose: bool = False) -> "Decoder55":
        s = renorm(s)
        best = self._obj(s, gold)
        for sw in range(sweeps):
            improved = False
            # biases
            for d in range(6):
                ks = (1, 2, 3) if d in FULL_DIMS else (3,)
                for k in ks:
                    keep = self.bias[d, k]
                    for v in BIAS_GRID:
                        self.bias[d, k] = v
                        o = self._obj(s, gold)
                        if o > best + 1e-9:
                            best, keep, improved = o, v, True
                    self.bias[d, k] = keep
            # P0 floors
            for d in range(6):
                keep = self.floor[d]
                for v in FLOOR_GRID:
                    self.floor[d] = v
                    o = self._obj(s, gold)
                    if o > best + 1e-9:
                        best, keep, improved = o, v, True
                self.floor[d] = keep
            # clean gate
            keep = self.t_clean
            for v in GATE_GRID:
                self.t_clean = v
                o = self._obj(s, gold)
                if o > best + 1e-9:
                    best, keep, improved = o, v, True
            self.t_clean = keep
            if verbose:
                print(f"    sweep {sw}: {self.objective}={best:.3f}")
            if not improved:
                break
        return self

    def state(self) -> Dict[str, object]:
        return {"objective": self.objective, "bias": self.bias.tolist(), "t_clean": float(self.t_clean), "floor": self.floor.tolist()}

    @classmethod
    def from_state(cls, st: Dict[str, object]) -> "Decoder55":
        d = cls(st["objective"])
        d.bias = np.asarray(st["bias"], dtype=np.float64)
        d.t_clean = float(st["t_clean"])
        d.floor = np.asarray(st["floor"], dtype=np.float64)
        return d


def nested_oof_decode(s: np.ndarray, gold: np.ndarray, fold: np.ndarray, objective: str = "doc1_final",
                      sweeps: int = 3) -> np.ndarray:
    """Leave-one-fold-out decoding: thresholds for fold f are fitted on the other 7 folds only."""
    pred = np.zeros_like(gold)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        dec = Decoder55(objective).fit(s[tr], gold[tr], sweeps=sweeps)
        pred[te] = dec.predict(s[te])
    return pred


def argmax_decode(s: np.ndarray) -> np.ndarray:
    return Decoder55().predict(s)
