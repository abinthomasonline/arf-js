import seedrandom from "seedrandom";
import type { RowRecord, Schema, TableData } from "../types";

interface NumericMarginal {
  kind: "numeric";
  min: number;
  max: number;
  mean: number;
  std: number;
}

interface CategoricalMarginal {
  kind: "categorical";
  values: unknown[];
  probabilities: number[];
}

export type ColumnMarginal = NumericMarginal | CategoricalMarginal;

export interface MarginalModel {
  schema: Schema;
  columns: Record<string, ColumnMarginal>;
}

type RNG = () => number;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

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

function fitNumeric(column: string, rows: RowRecord[]): NumericMarginal {
  const values = rows.map((row) => row[column]);
  const numericValues = values.map((value) => Number(value));

  if (numericValues.some((value) => Number.isNaN(value))) {
    throw new Error(`Column '${column}' includes non-numeric values.`);
  }

  const count = numericValues.length;
  const mean = numericValues.reduce((sum, value) => sum + value, 0) / count;
  const variance =
    numericValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(count, 1);
  const std = Math.sqrt(variance);

  return {
    kind: "numeric",
    min: Math.min(...numericValues),
    max: Math.max(...numericValues),
    mean,
    std,
  };
}

function fitCategorical(column: string, rows: RowRecord[]): CategoricalMarginal {
  const counts = new Map<unknown, number>();
  for (const row of rows) {
    const value = row[column];
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }

  const entries = [...counts.entries()];
  const total = rows.length;

  return {
    kind: "categorical",
    values: entries.map(([value]) => value),
    probabilities: entries.map(([, count]) => count / total),
  };
}

function sampleCategorical(marginal: CategoricalMarginal, rng: RNG): unknown {
  const target = rng();
  let cumulative = 0;

  for (let i = 0; i < marginal.values.length; i += 1) {
    cumulative += marginal.probabilities[i] ?? 0;
    if (target <= cumulative) {
      return marginal.values[i];
    }
  }

  return marginal.values[marginal.values.length - 1];
}

export function fitMarginals(input: TableData): MarginalModel {
  const columns: Record<string, ColumnMarginal> = {};

  for (const [column, kind] of Object.entries(input.schema)) {
    columns[column] = kind === "numeric" ? fitNumeric(column, input.rows) : fitCategorical(column, input.rows);
  }

  return { schema: input.schema, columns };
}

export function sampleIndependent(
  model: MarginalModel,
  count: number,
  seed: number | string = 42,
): RowRecord[] {
  if (count <= 0) {
    throw new Error("Sample count must be > 0.");
  }

  const rng = seedrandom(String(seed));
  const rows: RowRecord[] = [];

  for (let i = 0; i < count; i += 1) {
    const row: RowRecord = {};
    for (const column of Object.keys(model.schema)) {
      const marginal = model.columns[column];
      if (!marginal) {
        throw new Error(`Missing marginal for column '${column}'.`);
      }

      if (marginal.kind === "numeric") {
        const centered = marginal.std === 0 ? 0 : gaussian(rng) * marginal.std;
        row[column] = clamp(marginal.mean + centered, marginal.min, marginal.max);
      } else {
        row[column] = sampleCategorical(marginal, rng);
      }
    }
    rows.push(row);
  }

  return rows;
}
