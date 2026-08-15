from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_results import exact_sign_flip_pvalue, holm_adjust


def test_exact_sign_flip_test_has_expected_five_pair_resolution():
    differences = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert exact_sign_flip_pvalue(differences) == 0.0625


def test_holm_adjustment_preserves_original_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
