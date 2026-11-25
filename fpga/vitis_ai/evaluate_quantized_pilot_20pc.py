"""Evaluate the 20-per-category calibration on the unchanged pilot images/banks."""

from pathlib import Path

import evaluate_quantized_pilot as pilot

pilot.QUANT_DIR = Path("/out/quantize_result_packed_20pc")
pilot.OUT = Path("/out/evaluation_pilot_scores_20pc.json")

if __name__ == "__main__":
    pilot.main()
