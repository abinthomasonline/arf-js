export type ColumnKind = "numeric" | "categorical";

export type Schema = Record<string, ColumnKind>;

export type RowRecord = Record<string, unknown>;

export interface TableData {
  rows: RowRecord[];
  schema: Schema;
}

export interface DanfoTableData {
  dataframe: unknown;
  schema: Schema;
}

export type FitInput = TableData | DanfoTableData;

export interface ColumnMetadata {
  name: string;
  kind: ColumnKind;
  index: number;
}

export interface NormalizedTableData extends TableData {
  columns: ColumnMetadata[];
}

export interface FitOptions {
  seed?: number;
  delta?: number;
  maxIterations?: number;
  missingPolicy?: "reject" | "impute";
  numTrees?: number;
  minNodeSize?: number;
  maxFeatures?: number;
  gainThreshold?: number;
  laplaceAlpha?: number;
  earlyStop?: boolean;
  useOobAccuracy?: boolean;
}

export interface ArfIterationMetrics {
  iteration: number;
  accuracy: number;
  threshold: number;
  converged: boolean;
}

export interface NumericLeafParams {
  kind: "numeric";
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface CategoricalLeafParams {
  kind: "categorical";
  values: unknown[];
  probabilities: number[];
}

export type LeafFeatureParams = NumericLeafParams | CategoricalLeafParams;

export interface LeafParameters {
  path: string;
  count: number;
  coverage: number;
  features: Record<string, LeafFeatureParams>;
}

export interface TreeLeafParameters {
  treeIndex: number;
  leaves: LeafParameters[];
}

export interface LeafFitModel {
  trees: TreeLeafParameters[];
}

export interface ArfTrainingSummary {
  history: ArfIterationMetrics[];
  terminationReason: "converged" | "max_iterations";
  delta: number;
  maxIterations: number;
  leafModel?: LeafFitModel;
  leafSummary?: {
    treeCount: number;
    totalLeafCount: number;
    coverageSums: number[];
  };
}

export interface ArfModel {
  schema: Schema;
  fitted: boolean;
  metadata: {
    rowCount: number;
    seed: number;
    training: ArfTrainingSummary;
  };
}
