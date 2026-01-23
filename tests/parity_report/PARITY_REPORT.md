# Cairo vs diffvg-torch Parity Report

## Summary

| Test Case | MAE | RGB MAE | PSNR | Status |
|-----------|-----|---------|------|--------|
| Simple Triangle | 0.5 | 0.6 | 35.7 | PASS |
| Red Rectangle | 0.2 | 0.3 | 40.4 | PASS |
| Blue Circle | 0.4 | 0.5 | 35.2 | PASS |
| Cubic Bezier | 0.5 | 0.6 | 31.9 | PASS |
| Two Overlapping | 0.7 | 0.9 | 32.6 | PASS |
| Star Shape | 0.2 | 0.3 | 40.1 | PASS |
| Three Colors | 0.8 | 1.0 | 33.4 | PASS |
| Fill + Stroke | 0.7 | 1.0 | 33.1 | PASS |

## Notes

- MAE: Mean Absolute Error (0-255 scale)
- RGB MAE: MAE computed only on RGB channels
- PSNR: Peak Signal-to-Noise Ratio (higher is better)
- PASS: RGB MAE < 35
- WARN: RGB MAE 35-50
- FAIL: RGB MAE > 50
