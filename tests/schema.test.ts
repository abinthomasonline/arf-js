import { describe, expect, it } from "vitest";
import { validateRowsAgainstSchema, validateSchema } from "../src/core/schema";
import type { Schema } from "../src/core/types";

describe("schema validation", () => {
  it("accepts numeric and categorical schema", () => {
    const schema: Schema = {
      age: "numeric",
      city: "categorical",
    };

    expect(() => validateSchema(schema)).not.toThrow();
  });

  it("rejects unsupported schema type", () => {
    expect(() =>
      validateSchema({
        age: "numeric",
        // @ts-expect-error test runtime invalid kind
        active: "boolean",
      }),
    ).toThrow("Unsupported column kind");
  });

  it("rejects rows missing required columns", () => {
    const schema: Schema = { age: "numeric", city: "categorical" };
    expect(() => validateRowsAgainstSchema([{ age: 30 }], schema)).toThrow(
      "missing required column",
    );
  });
});
