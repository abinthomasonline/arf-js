import { describe, expect, it } from "vitest";
import { fitLeafParameters, summarizeLeafCoverage } from "../src/core/leaves";
import { MlRandomForestAdapter } from "../src/core/rf";
import type { ForestSnapshot } from "../src/core/rf";

describe("leaf fitting", () => {
  it("fits per-leaf params and normalized coverage", () => {
    const schema = { age: "numeric", city: "categorical" } as const;
    const realRows = [
      { age: 20, city: "A" },
      { age: 30, city: "B" },
      { age: 40, city: "A" },
      { age: 50, city: "C" },
      { age: 60, city: "B" },
      { age: 70, city: "C" },
    ];

    const realEncoded = [
      [20, 0],
      [30, 1],
      [40, 0],
      [50, 2],
      [60, 1],
      [70, 2],
    ];
    const syntheticEncoded = [
      [22, 0],
      [28, 1],
      [35, 2],
      [58, 0],
      [64, 1],
      [68, 2],
    ];

    const adapter = new MlRandomForestAdapter({ seed: 11, nEstimators: 5 });
    adapter.fit({
      features: [...realEncoded, ...syntheticEncoded],
      labels: [...new Array(realEncoded.length).fill(1), ...new Array(syntheticEncoded.length).fill(0)],
    });

    const model = fitLeafParameters({
      rows: realRows,
      encodedRows: realEncoded,
      schema,
      forest: adapter.getForestSnapshot(),
    });

    const summary = summarizeLeafCoverage(model);
    expect(summary.length).toBeGreaterThan(0);
    expect(summary.every((s) => s.coverageSum > 0.999 && s.coverageSum < 1.001)).toBe(true);
    expect(summary.every((s) => s.leafCount > 0)).toBe(true);

    for (const tree of model.trees) {
      for (const leaf of tree.leaves) {
        const age = leaf.features.age;
        const city = leaf.features.city;

        if (!age || !city) {
          throw new Error("Leaf is missing expected feature params.");
        }

        expect(age.kind).toBe("numeric");
        if (age.kind === "numeric") {
          expect(age.min).toBeLessThanOrEqual(age.max);
          expect(age.std).toBeGreaterThanOrEqual(0);
        }

        expect(city.kind).toBe("categorical");
        if (city.kind === "categorical") {
          const probSum = city.probabilities.reduce((sum: number, p: number) => sum + p, 0);
          expect(city.values.length).toBeGreaterThan(0);
          expect(probSum).toBeCloseTo(1, 8);
        }
      }
    }
  });

  it("uses split bounds for numeric min/max and Laplace smoothing for categorical", () => {
    const schema = { x: "numeric", cat: "categorical" } as const;
    const rows = [
      { x: 0.2, cat: "A" },
      { x: 0.3, cat: "A" },
      { x: 0.8, cat: "C" },
      { x: 0.9, cat: "C" },
    ];
    const encodedRows = [
      [0.2, 0],
      [0.3, 0],
      [0.8, 2],
      [0.9, 2],
    ];

    const forest: ForestSnapshot = {
      trees: [
        {
          root: {
            splitColumn: 0,
            splitValue: 0.5,
            left: {},
            right: {},
          },
        },
      ],
      featureIndexes: [[0, 1]],
    };

    const model = fitLeafParameters({
      rows,
      encodedRows,
      schema,
      forest,
      categoryCodebooks: { cat: ["A", "B", "C"] },
      laplaceAlpha: 1,
    });

    const leftLeaf = model.trees[0]?.leaves.find((leaf) => leaf.path === "L");
    expect(leftLeaf).toBeTruthy();
    if (!leftLeaf) {
      return;
    }

    const xParams = leftLeaf.features.x;
    const catParams = leftLeaf.features.cat;

    expect(xParams?.kind).toBe("numeric");
    if (xParams?.kind === "numeric") {
      expect(xParams.min).toBe(-Infinity);
      expect(xParams.max).toBe(0.5);
      expect(xParams.mean).toBeCloseTo(0.25, 8);
    }

    expect(catParams?.kind).toBe("categorical");
    if (catParams?.kind === "categorical") {
      expect(catParams.values).toEqual(["A", "B", "C"]);
      const probs = catParams.probabilities;
      expect(probs[0]).toBeCloseTo(0.6, 8);
      expect(probs[1]).toBeCloseTo(0.2, 8);
      expect(probs[2]).toBeCloseTo(0.2, 8);
    }
  });
});
