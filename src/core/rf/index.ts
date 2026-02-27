import { RandomForestClassifier } from "ml-random-forest";

export interface RfTrainingInput {
  features: number[][];
  labels: number[];
}

export interface RfAdapterOptions {
  nEstimators?: number;
  maxFeatures?: number;
  replacement?: boolean;
  seed?: number;
  noOOB?: boolean;
  treeOptions?: {
    maxDepth?: number;
    minNumSamples?: number;
    gainThreshold?: number;
  };
}

export interface ForestSnapshot {
  trees: unknown[];
  featureIndexes: number[][];
}

export interface RfAdapter {
  fit(input: RfTrainingInput): void;
  predict(features: number[][]): number[];
  getForestSnapshot(): ForestSnapshot;
  getOobAccuracy(): number | null;
}

function assert2D(features: number[][], fieldName: string): void {
  if (!Array.isArray(features) || features.length === 0) {
    throw new Error(`${fieldName} must be a non-empty 2D array.`);
  }

  const width = features[0]?.length ?? 0;
  if (width === 0) {
    throw new Error(`${fieldName} rows must contain at least one feature.`);
  }

  for (let i = 0; i < features.length; i += 1) {
    const row = features[i];
    if (!row || row.length !== width) {
      throw new Error(`${fieldName} must be rectangular.`);
    }
  }
}

export class MlRandomForestAdapter implements RfAdapter {
  private readonly options: RfAdapterOptions;
  private model: RandomForestClassifier | null = null;
  private hasOob = false;

  constructor(options: RfAdapterOptions = {}) {
    this.options = options;
  }

  fit(input: RfTrainingInput): void {
    assert2D(input.features, "features");

    if (input.labels.length !== input.features.length) {
      throw new Error("labels length must match feature row count.");
    }

    const featureCount = input.features[0]?.length ?? 1;
    const inferredMaxFeatures = Math.min(Math.max(Math.sqrt(featureCount) / featureCount, 0.0001), 1);
    const treeOptions: { minNumSamples: number; gainThreshold: number; maxDepth?: number } = {
      minNumSamples: this.options.treeOptions?.minNumSamples ?? 5,
      gainThreshold: this.options.treeOptions?.gainThreshold ?? 0,
    };
    if (this.options.treeOptions?.maxDepth !== undefined) {
      treeOptions.maxDepth = this.options.treeOptions.maxDepth;
    }

    const trainWithNoOob = (noOOB: boolean): RandomForestClassifier => {
      const model = new RandomForestClassifier({
        nEstimators: this.options.nEstimators ?? 30,
        maxFeatures: this.options.maxFeatures ?? inferredMaxFeatures,
        replacement: this.options.replacement ?? true,
        seed: this.options.seed,
        noOOB,
        treeOptions,
      });
      model.train(input.features, input.labels);
      return model;
    };

    const requestedNoOob = this.options.noOOB ?? false;
    try {
      this.model = trainWithNoOob(requestedNoOob);
      this.hasOob = !requestedNoOob;
    } catch (error) {
      if (!requestedNoOob) {
        // Fallback for small-sample OOB edge cases in ml-random-forest.
        this.model = trainWithNoOob(true);
        this.hasOob = false;
      } else {
        throw error;
      }
    }
  }

  predict(features: number[][]): number[] {
    assert2D(features, "features");

    if (!this.model) {
      throw new Error("RF model is not fitted.");
    }

    return this.model.predict(features).map((value) => Number(value));
  }

  getForestSnapshot(): ForestSnapshot {
    if (!this.model) {
      throw new Error("RF model is not fitted.");
    }

    return {
      trees: [...(this.model.estimators ?? [])],
      featureIndexes: (this.model.indexes ?? []).map((indexes) => [...(indexes ?? [])]),
    };
  }

  getOobAccuracy(): number | null {
    if (!this.model) {
      throw new Error("RF model is not fitted.");
    }

    const oob = (this.model as unknown as { oobResults?: Array<{ true: number; predicted: number }> })
      .oobResults;
    if (!this.hasOob || !oob || oob.length === 0) {
      return null;
    }

    let correct = 0;
    for (const row of oob) {
      if (row.true === row.predicted) {
        correct += 1;
      }
    }
    return correct / oob.length;
  }
}
