import { describe, expect, it } from "vitest";
import { fitMarginals, sampleIndependent } from "../src/core/marginals";

describe("marginals", () => {
  const input = {
    schema: { age: "numeric", city: "categorical" } as const,
    rows: [
      { age: 20, city: "A" },
      { age: 30, city: "B" },
      { age: 40, city: "A" },
      { age: 50, city: "C" },
    ],
  };

  it("fits numeric and categorical marginals", () => {
    const model = fitMarginals(input);

    expect(model.columns.age?.kind).toBe("numeric");
    expect(model.columns.city?.kind).toBe("categorical");
  });

  it("samples rows with matching schema and type classes", () => {
    const model = fitMarginals(input);
    const rows = sampleIndependent(model, 10, 123);

    expect(rows).toHaveLength(10);
    for (const row of rows) {
      expect(typeof row.age).toBe("number");
      expect(["A", "B", "C"]).toContain(row.city);
    }
  });

  it("is reproducible with fixed seed", () => {
    const model = fitMarginals(input);

    const sampleA = sampleIndependent(model, 5, 999);
    const sampleB = sampleIndependent(model, 5, 999);

    expect(sampleA).toEqual(sampleB);
  });
});
