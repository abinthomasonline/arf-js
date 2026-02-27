# arf-js

ARF (Adversarial Random Forest) core library in TypeScript for synthetic tabular data generation.

## Scope

This repository currently contains only the ARF core:
- train ARF with explicit schema (`numeric` / `categorical`)
- sample synthetic rows from trained model
- deterministic behavior via seed

Out of scope (for now):
- preprocessing/transformer layer
- web UI

## Install

```bash
npm install
```

## API

```ts
import { fit, sample, fitSample } from "arf-js";
```

- `fit({ rows, schema }, options?) -> model`
- `sample(model, n, seed?) -> rows`
- `fitSample({ rows, schema }, n, options?) -> rows`

## Data contract

- Input rows are plain records.
- Schema is required and must map each column to `"numeric"` or `"categorical"`.
- Missing handling is controlled by `missingPolicy` in fit options:
  - `"reject"` (default)
  - `"impute"`

## Development

```bash
npm run typecheck
npm test
npm run build
```

## Examples

```bash
npm run examples:twomoons
npm run examples:digits
npm run examples:mvnorm
npm run examples:titanic
npm run examples:penguins
npm run examples:all
```

See [examples/README.md](examples/README.md) for details.
