import * as dfd from "danfojs";
import { validateRowsAgainstSchema, validateSchema } from "../schema";
import type {
  ColumnMetadata,
  DanfoTableData,
  FitInput,
  NormalizedTableData,
  RowRecord,
  Schema,
  TableData,
} from "../types";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isDanfoTableData(input: FitInput): input is DanfoTableData {
  return isObject(input) && "dataframe" in input && "schema" in input;
}

export function buildColumnMetadata(schema: Schema): ColumnMetadata[] {
  return Object.entries(schema).map(([name, kind], index) => ({
    name,
    kind,
    index,
  }));
}

function normalizeRows(rows: RowRecord[], schema: Schema): RowRecord[] {
  const orderedColumns = Object.keys(schema);

  return rows.map((row) => {
    const normalized: RowRecord = {};
    for (const column of orderedColumns) {
      normalized[column] = row[column];
    }
    return normalized;
  });
}

function rowsFromDanfoDataFrame(dataframe: unknown): RowRecord[] {
  const raw = (dfd as { toJSON: (frame: unknown, options: { format: "column" }) => unknown }).toJSON(
    dataframe,
    { format: "column" },
  );

  if (!Array.isArray(raw)) {
    throw new Error("Failed to convert danfojs dataframe to row records.");
  }

  return raw as RowRecord[];
}

export function normalizeTableInput(input: FitInput): NormalizedTableData {
  const tableData: TableData = isDanfoTableData(input)
    ? { rows: rowsFromDanfoDataFrame(input.dataframe), schema: input.schema }
    : input;

  validateSchema(tableData.schema);
  validateRowsAgainstSchema(tableData.rows, tableData.schema);

  return {
    rows: normalizeRows(tableData.rows, tableData.schema),
    schema: tableData.schema,
    columns: buildColumnMetadata(tableData.schema),
  };
}
