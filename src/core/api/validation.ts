import type { FitOptions, RowRecord, Schema } from "../types";

function isMissing(value: unknown): boolean {
  return value === null || value === undefined || (typeof value === "number" && Number.isNaN(value));
}

export function validateFitOptions(options: FitOptions): void {
  if (options.delta !== undefined) {
    if (!Number.isFinite(options.delta) || options.delta < 0 || options.delta > 0.5) {
      throw new Error("Invalid fit option 'delta': expected a finite number in [0, 0.5].");
    }
  }

  if (options.maxIterations !== undefined) {
    if (!Number.isInteger(options.maxIterations) || options.maxIterations <= 0) {
      throw new Error("Invalid fit option 'maxIterations': expected a positive integer.");
    }
  }

  if (options.seed !== undefined) {
    if (!Number.isInteger(options.seed)) {
      throw new Error("Invalid fit option 'seed': expected an integer.");
    }
  }

  if (options.numTrees !== undefined) {
    if (!Number.isInteger(options.numTrees) || options.numTrees <= 0) {
      throw new Error("Invalid fit option 'numTrees': expected a positive integer.");
    }
  }

  if (options.minNodeSize !== undefined) {
    if (!Number.isInteger(options.minNodeSize) || options.minNodeSize <= 0) {
      throw new Error("Invalid fit option 'minNodeSize': expected a positive integer.");
    }
  }

  if (options.maxFeatures !== undefined) {
    if (!Number.isFinite(options.maxFeatures) || options.maxFeatures <= 0) {
      throw new Error("Invalid fit option 'maxFeatures': expected a positive number.");
    }
  }

  if (options.gainThreshold !== undefined) {
    if (!Number.isFinite(options.gainThreshold) || options.gainThreshold < 0) {
      throw new Error("Invalid fit option 'gainThreshold': expected a non-negative number.");
    }
  }

  if (options.laplaceAlpha !== undefined) {
    if (!Number.isFinite(options.laplaceAlpha) || options.laplaceAlpha < 0) {
      throw new Error("Invalid fit option 'laplaceAlpha': expected a non-negative number.");
    }
  }

  if (options.earlyStop !== undefined && typeof options.earlyStop !== "boolean") {
    throw new Error("Invalid fit option 'earlyStop': expected a boolean.");
  }

  if (options.useOobAccuracy !== undefined && typeof options.useOobAccuracy !== "boolean") {
    throw new Error("Invalid fit option 'useOobAccuracy': expected a boolean.");
  }

  if (options.missingPolicy !== undefined && options.missingPolicy !== "reject" && options.missingPolicy !== "impute") {
    throw new Error("Invalid fit option 'missingPolicy': expected 'reject' or 'impute'.");
  }
}

function imputeNumeric(column: string, rows: RowRecord[]): number {
  const values = rows
    .map((row) => row[column])
    .filter((value) => !isMissing(value))
    .map((value) => Number(value));

  if (values.length === 0 || values.some((value) => Number.isNaN(value))) {
    throw new Error(`Cannot impute numeric column '${column}': no valid numeric values.`);
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function imputeCategorical(column: string, rows: RowRecord[]): unknown {
  const counts = new Map<unknown, number>();

  for (const row of rows) {
    const value = row[column];
    if (!isMissing(value)) {
      counts.set(value, (counts.get(value) ?? 0) + 1);
    }
  }

  if (counts.size === 0) {
    throw new Error(`Cannot impute categorical column '${column}': no valid observed values.`);
  }

  let bestValue: unknown = undefined;
  let bestCount = -1;
  for (const [value, count] of counts.entries()) {
    if (count > bestCount) {
      bestValue = value;
      bestCount = count;
    }
  }

  return bestValue;
}

export function applyMissingPolicy(
  rows: RowRecord[],
  schema: Schema,
  policy: FitOptions["missingPolicy"] = "reject",
): RowRecord[] {
  const effectivePolicy = policy ?? "reject";

  if (effectivePolicy === "reject") {
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
      const row = rows[rowIndex];
      if (!row) {
        throw new Error(`Row ${rowIndex} is missing.`);
      }

      for (const column of Object.keys(schema)) {
        if (isMissing(row[column])) {
          throw new Error(
            `Missing value found at row ${rowIndex}, column '${column}'. Set missingPolicy='impute' or clean input before fit.`,
          );
        }
      }
    }

    return rows;
  }

  const numericImputations: Record<string, number> = {};
  const categoricalImputations: Record<string, unknown> = {};

  for (const [column, kind] of Object.entries(schema)) {
    if (kind === "numeric") {
      numericImputations[column] = imputeNumeric(column, rows);
    } else {
      categoricalImputations[column] = imputeCategorical(column, rows);
    }
  }

  return rows.map((row) => {
    const nextRow: RowRecord = { ...row };
    for (const [column, kind] of Object.entries(schema)) {
      if (isMissing(nextRow[column])) {
        nextRow[column] = kind === "numeric" ? numericImputations[column] : categoricalImputations[column];
      }
    }
    return nextRow;
  });
}
