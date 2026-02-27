import type { RowRecord, Schema } from "../types";

export function validateSchema(schema: Schema): void {
  const entries = Object.entries(schema);
  if (entries.length === 0) {
    throw new Error("Schema must include at least one column.");
  }

  for (const [column, kind] of entries) {
    if (!column) {
      throw new Error("Schema contains an empty column name.");
    }
    if (kind !== "numeric" && kind !== "categorical") {
      throw new Error(
        `Unsupported column kind for '${column}': ${String(kind)}. Expected numeric or categorical.`,
      );
    }
  }
}

export function validateRowsAgainstSchema(rows: RowRecord[], schema: Schema): void {
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    if (!row) {
      throw new Error(`Row ${i} is missing.`);
    }
    for (const column of Object.keys(schema)) {
      if (!(column in row)) {
        throw new Error(`Row ${i} is missing required column '${column}'.`);
      }
    }
  }
}
