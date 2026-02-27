import { describe, expect, it } from "vitest";
import { fit, sample } from "../src/core/api";

const cleanInput = {
  schema: { age: "numeric", city: "categorical" } as const,
  rows: [
    { age: 20, city: "A" },
    { age: 30, city: "B" },
    { age: 40, city: "A" },
    { age: 50, city: "C" },
  ],
};

describe("api validation", () => {
  it("rejects invalid fit options", () => {
    expect(() => fit(cleanInput, { delta: -0.1 })).toThrow("Invalid fit option 'delta'");
    expect(() => fit(cleanInput, { delta: 0.9 })).toThrow("Invalid fit option 'delta'");
    expect(() => fit(cleanInput, { maxIterations: 0 })).toThrow("Invalid fit option 'maxIterations'");
    expect(() => fit(cleanInput, { seed: 1.2 })).toThrow("Invalid fit option 'seed'");
    expect(() => fit(cleanInput, { gainThreshold: -1 })).toThrow("Invalid fit option 'gainThreshold'");
    expect(() => fit(cleanInput, { laplaceAlpha: -1 })).toThrow("Invalid fit option 'laplaceAlpha'");
  });

  it("rejects missing values by default", () => {
    const input = {
      schema: { age: "numeric", city: "categorical" } as const,
      rows: [
        { age: 20, city: "A" },
        { age: undefined, city: "B" },
      ],
    };

    expect(() => fit(input)).toThrow("Missing value found");
  });

  it("supports missingPolicy=impute", () => {
    const input = {
      schema: { age: "numeric", city: "categorical" } as const,
      rows: [
        { age: 20, city: "A" },
        { age: undefined, city: "B" },
        { age: 40, city: undefined },
      ],
    };

    const model = fit(input, { missingPolicy: "impute", maxIterations: 2, seed: 12 });
    expect(model.fitted).toBe(true);
    expect(model.metadata.training.history.length).toBeGreaterThan(0);
  });

  it("rejects invalid sample size", () => {
    const model = fit(cleanInput, { seed: 5, maxIterations: 2 });
    expect(() => sample(model, 0)).toThrow("positive integer");
    expect(() => sample(model, 1.5)).toThrow("positive integer");
  });
});
