import seedrandom from "seedrandom";
import type { ArfModel, CategoricalLeafParams, LeafParameters, NumericLeafParams, RowRecord } from "../types";

type RNG = () => number;

function gaussian(rng: RNG): number {
  let u1 = 0;
  let u2 = 0;
  while (u1 === 0) {
    u1 = rng();
  }
  while (u2 === 0) {
    u2 = rng();
  }
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function weightedPick<T>(items: T[], weights: number[], rng: RNG): T {
  const target = rng();
  let cumulative = 0;
  for (let i = 0; i < items.length; i += 1) {
    cumulative += weights[i] ?? 0;
    if (target <= cumulative) {
      const item = items[i];
      if (item === undefined) {
        throw new Error("Invalid weighted selection.");
      }
      return item;
    }
  }
  const fallback = items[items.length - 1];
  if (fallback === undefined) {
    throw new Error("Cannot select from an empty collection.");
  }
  return fallback;
}

function sampleNumeric(params: NumericLeafParams, rng: RNG): number {
  const centered = params.std === 0 ? 0 : gaussian(rng) * params.std;
  return clamp(params.mean + centered, params.min, params.max);
}

function sampleCategorical(params: CategoricalLeafParams, rng: RNG): unknown {
  return weightedPick(params.values, params.probabilities, rng);
}

function sampleRowFromLeaf(leaf: LeafParameters, schemaColumns: string[], rng: RNG): RowRecord {
  const row: RowRecord = {};
  for (const column of schemaColumns) {
    const params = leaf.features[column];
    if (!params) {
      throw new Error(`Missing leaf params for column '${column}'.`);
    }
    row[column] = params.kind === "numeric" ? sampleNumeric(params, rng) : sampleCategorical(params, rng);
  }
  return row;
}

export function sampleSynthetic(model: ArfModel, n: number, seed: number | string = 42): RowRecord[] {
  const leafModel = model.metadata.training.leafModel;
  if (!leafModel || leafModel.trees.length === 0) {
    throw new Error("Model does not include fitted leaf distributions.");
  }

  const rng = seedrandom(String(seed));
  const rows: RowRecord[] = [];
  const schemaColumns = Object.keys(model.schema);

  for (let i = 0; i < n; i += 1) {
    const treeWeights = new Array(leafModel.trees.length).fill(1 / leafModel.trees.length);
    const tree = weightedPick(leafModel.trees, treeWeights, rng);
    if (tree.leaves.length === 0) {
      throw new Error(`Tree ${tree.treeIndex} has no leaves.`);
    }

    const leaf = weightedPick(
      tree.leaves,
      tree.leaves.map((item) => item.coverage),
      rng,
    );

    rows.push(sampleRowFromLeaf(leaf, schemaColumns, rng));
  }

  return rows;
}
