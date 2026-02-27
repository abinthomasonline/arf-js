import { runArfTraining } from "../arf";
import { normalizeTableInput } from "../table";
import { sampleSynthetic } from "../forge";
import type { ArfModel, FitInput, FitOptions, RowRecord } from "../types";
import { applyMissingPolicy, validateFitOptions } from "./validation";

export function fit(input: FitInput, options: FitOptions = {}): ArfModel {
  validateFitOptions(options);
  const normalizedInput = normalizeTableInput(input);
  const rows = applyMissingPolicy(normalizedInput.rows, normalizedInput.schema, options.missingPolicy);
  return runArfTraining({ ...normalizedInput, rows }, options);
}

export function sample(model: ArfModel, n: number, seed?: number | string): RowRecord[] {
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error("Sample size must be a positive integer.");
  }
  return sampleSynthetic(model, n, seed ?? model.metadata.seed);
}

export function fitSample(input: FitInput, n: number, options: FitOptions = {}): RowRecord[] {
  const model = fit(input, options);
  return sample(model, n);
}
