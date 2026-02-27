import { describe, expect, it } from "vitest";
import { MlRandomForestAdapter } from "../src/core/rf";

describe("ml-random-forest adapter", () => {
  it("trains and predicts with expected shape", () => {
    const adapter = new MlRandomForestAdapter({ seed: 42, nEstimators: 50, noOOB: true });

    const features = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    const labels = [0, 1, 1, 1];

    adapter.fit({ features, labels });
    const predictions = adapter.predict(features);

    expect(predictions).toHaveLength(features.length);
    for (const value of predictions) {
      expect([0, 1]).toContain(value);
    }
  });

  it("is deterministic with the same seed and data", () => {
    const features = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
      [0, 2],
      [2, 0],
    ];
    const labels = [0, 1, 1, 1, 1, 1];

    const first = new MlRandomForestAdapter({ seed: 7, nEstimators: 40, noOOB: true });
    const second = new MlRandomForestAdapter({ seed: 7, nEstimators: 40, noOOB: true });

    first.fit({ features, labels });
    second.fit({ features, labels });

    expect(first.predict(features)).toEqual(second.predict(features));
  });

  it("rejects predict before fit", () => {
    const adapter = new MlRandomForestAdapter();
    expect(() => adapter.predict([[1, 2]])).toThrow("not fitted");
  });
});
