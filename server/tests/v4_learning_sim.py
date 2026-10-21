"""A v4 simulator whose boards really learn: the int16 hardware twin's local training on the
machine data of each position, firmware-style float32 quantisation of the delta, and a real
REPLAY pass with the final model. Used to run analysis/compare_hw_twin_v4.py on run directories
produced by the server.

Importing this module needs the research folder and its int16 caches (like the analysis tool).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import compare_hw_twin_v4 as cmp  # noqa: E402  (the unchanged analysis tool)
import synth_v4_run as synth      # noqa: E402  (its firmware-arithmetic helpers)

from protocol_v4 import QuantisedModel  # noqa: E402
from v4_sim import Sim  # noqa: E402

twin = cmp.twin
F32 = np.float32


def load_machine_data():
    """(data, evaluator) of the analysis tool, or raise FileNotFoundError / KeyError if caches are absent."""
    return cmp.load_data()


class LearningSim(Sim):
    def __init__(self, *args, data, **kwargs) -> None:
        self.data = data
        super().__init__(*args, **kwargs)

    def _board_params(self, global_model):
        """The model a board decodes from a GLOBAL: float32 multiply, as the firmware does."""
        return synth.unflat(synth.dequantize(global_model.model.codes, global_model.model.scales, F32))

    def make_delta(self, board, position, phase_index, round_, base_model):
        source = self._board_params(base_model)
        _, arrays, x = self.data[position]
        trained = twin.local_train(x[arrays["train_indices"].astype(int)], source)
        delta = tuple(np.asarray(t, F32) - np.asarray(s, F32) for t, s in zip(trained, source))
        codes, scales = synth.fw_quantize_update(delta)
        return QuantisedModel(codes=codes, scales=tuple(float(s) for s in scales)), 0.01

    def replay_result(self, board, position, phase_index, final_model):
        flags = cmp.flags_for(self._board_params(final_model), self.data[position])
        hit = np.flatnonzero(flags)
        return len(flags), (int(hit[0]) if len(hit) else None)
