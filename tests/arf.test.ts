import { describe, expect, it } from "vitest";
import { runArfTraining } from "../src/core/arf";
import type { ForestSnapshot, RfAdapter, RfTrainingInput } from "../src/core/rf";

class ScriptedAdapter implements RfAdapter {
  private labels: number[] = [];
  private readonly targetAccuracy: number;

  constructor(targetAccuracy: number) {
    this.targetAccuracy = targetAccuracy;
  }

  fit(input: RfTrainingInput): void {
    this.labels = input.labels;
  }

  predict(features: number[][]): number[] {
    const total = features.length;
    const keep = Math.round(this.targetAccuracy * total);

    return this.labels.map((label, index) => {
      if (index < keep) {
        return label;
      }
      return label === 1 ? 0 : 1;
    });
  }

  getForestSnapshot(): ForestSnapshot {
    return {
      trees: [{ root: {} }],
      featureIndexes: [[0, 1]],
    };
  }

  getOobAccuracy(): number | null {
    return null;
  }
}

describe("arf loop", () => {
  const input = {
    schema: { age: "numeric", city: "categorical" } as const,
    rows: [
      { age: 20, city: "A" },
      { age: 30, city: "B" },
      { age: 40, city: "A" },
      { age: 50, city: "C" },
    ],
  };

  it("terminates as converged when accuracy reaches threshold", () => {
    const model = runArfTraining(input, { delta: 0, maxIterations: 5 }, {
      createRfAdapter: () => new ScriptedAdapter(0.5),
    });

    expect(model.metadata.training.terminationReason).toBe("converged");
    expect(model.metadata.training.history).toHaveLength(1);
    expect(model.metadata.training.history[0]?.converged).toBe(true);
    expect(model.metadata.training.leafSummary?.treeCount).toBe(1);
  });

  it("terminates by max iterations when accuracy stays high", () => {
    const model = runArfTraining(input, { delta: 0, maxIterations: 3 }, {
      createRfAdapter: () => new ScriptedAdapter(0.9),
    });

    expect(model.metadata.training.terminationReason).toBe("max_iterations");
    expect(model.metadata.training.history).toHaveLength(4);
    expect(model.metadata.training.history.every((item) => item.converged === false)).toBe(true);
    expect(model.metadata.training.leafSummary?.coverageSums[0]).toBeCloseTo(1, 6);
  });
});
