import * as dfd from "danfojs";
import { describe, expect, it } from "vitest";
import { buildColumnMetadata, normalizeTableInput } from "../src/core/table";
import type { Schema } from "../src/core/types";

describe("table adapter", () => {
  it("builds stable column metadata in schema order", () => {
    const schema: Schema = {
      city: "categorical",
      age: "numeric",
    };

    const metadata = buildColumnMetadata(schema);
    expect(metadata).toEqual([
      { name: "city", kind: "categorical", index: 0 },
      { name: "age", kind: "numeric", index: 1 },
    ]);
  });

  it("normalizes row-record input to schema column order", () => {
    const normalized = normalizeTableInput({
      schema: { city: "categorical", age: "numeric" },
      rows: [{ age: 30, city: "A" }],
    });

    expect(Object.keys(normalized.rows[0] ?? {})).toEqual(["city", "age"]);
    expect(normalized.columns.map((col) => col.name)).toEqual(["city", "age"]);
  });

  it("accepts danfojs dataframe input", () => {
    const dataframe = new dfd.DataFrame([
      { age: 30, city: "A" },
      { age: 41, city: "B" },
    ]);

    const normalized = normalizeTableInput({
      dataframe,
      schema: { age: "numeric", city: "categorical" },
    });

    expect(normalized.rows).toHaveLength(2);
    expect(normalized.rows[0]).toEqual({ age: 30, city: "A" });
  });
});
