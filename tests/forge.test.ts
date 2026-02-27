import { describe, expect, it } from "vitest";
import { fit, sample } from "../src/core/api";

describe("forge sampling", () => {
  const input = {
    schema: { age: "numeric", city: "categorical" } as const,
    rows: [
      { age: 20, city: "A" },
      { age: 30, city: "B" },
      { age: 40, city: "A" },
      { age: 50, city: "C" },
      { age: 60, city: "B" },
      { age: 70, city: "C" },
    ],
  };

  it("generates requested row count with schema-compatible values", () => {
    const model = fit(input, { seed: 9, maxIterations: 3 });
    const rows = sample(model, 12, 9);

    expect(rows).toHaveLength(12);
    for (const row of rows) {
      expect(Object.keys(row)).toEqual(["age", "city"]);
      expect(typeof row.age).toBe("number");
      expect(["A", "B", "C"]).toContain(row.city);
    }
  });

  it("is deterministic for the same seed", () => {
    const model = fit(input, { seed: 11, maxIterations: 3 });

    const a = sample(model, 8, 777);
    const b = sample(model, 8, 777);

    expect(a).toEqual(b);
  });
});
