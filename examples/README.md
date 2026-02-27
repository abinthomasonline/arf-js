# Example Verification Workflows

This folder contains reproducible visual checks for the ARF core.

## Prerequisites
- JS deps installed (`npm install`)
- Python env `arf-js` with `scikit-learn`, `pandas`, `matplotlib`

## Two Moons
Runs train -> synth -> scatter plot with matched axis scales.

```bash
npm run examples:twomoons
```

Outputs:
- `examples/data/twomoons_train.json`
- `examples/output/twomoons_synth.json`
- `examples/output/twomoons_training.json`
- `examples/output/twomoons_scatter.png`

## Digits (zeros)
Runs train -> synth(4) -> image grid plot.

```bash
npm run examples:digits
```

Outputs:
- `examples/data/digits_zero_train.json`
- `examples/output/digits_synth.json`
- `examples/output/digits_training.json`
- `examples/output/digits_grid.png`

## MVNorm
Runs train -> synth -> scatter plot (matched axis scales), aligned to the arfpy `mvnorm` tutorial setup.

```bash
npm run examples:mvnorm
```

Outputs:
- `examples/data/mvnorm_train.json`
- `examples/output/mvnorm_synth.json`
- `examples/output/mvnorm_training.json`
- `examples/output/mvnorm_scatter.png`

## Notes
- Two-moons uses tutorial-like settings (`numTrees=20`, `minNodeSize=2`, `maxFeatures=2`).
- Digits follows arfpy tutorial idea (`load_digits(n_class=1)`, categorical pixels, synthesize 4 examples).
