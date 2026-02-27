import { describe, expect, it } from "vitest";
import { fitSample } from "../src/core/api";

describe("e2e smoke", () => {
  it("fitSample returns synthetic rows with same schema", () => {
    const input = {
      schema: { age: "numeric", city: "categorical" } as const,
      rows: [
        { age: 21, city: "A" },
        { age: 35, city: "B" },
        { age: 47, city: "A" },
        { age: 59, city: "C" },
        { age: 63, city: "B" },
      ],
    };

    const synthetic = fitSample(input, 7, { seed: 101, maxIterations: 3 });

    expect(synthetic).toHaveLength(7);
    for (const row of synthetic) {
      expect(Object.keys(row)).toEqual(["age", "city"]);
      expect(typeof row.age).toBe("number");
      expect(["A", "B", "C"]).toContain(row.city);
    }
  });
});
