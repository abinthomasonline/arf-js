import * as dfd from "danfojs";
import { describe, expect, it } from "vitest";
import { fit } from "../src/core/api";

describe("api fit", () => {
  it("returns scaffold model metadata", () => {
    const model = fit({
      schema: { age: "numeric", city: "categorical" },
      rows: [
        { age: 30, city: "A" },
        { age: 41, city: "B" },
      ],
    });

    expect(model.metadata.rowCount).toBe(2);
    expect(model.metadata.seed).toBe(42);
    expect(model.fitted).toBe(true);
    expect(model.metadata.training.history.length).toBeGreaterThan(0);
    expect(model.metadata.training.leafSummary?.treeCount).toBe(30);
  });

  it("accepts danfojs dataframe input", () => {
    const dataframe = new dfd.DataFrame([
      { age: 30, city: "A" },
      { age: 41, city: "B" },
    ]);

    const model = fit({
      dataframe,
      schema: { age: "numeric", city: "categorical" },
    });

    expect(model.metadata.rowCount).toBe(2);
    expect(model.metadata.training.history.length).toBeGreaterThan(0);
  });
});
