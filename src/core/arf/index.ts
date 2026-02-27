import seedrandom from "seedrandom";
import { fitLeafParameters, summarizeLeafCoverage } from "../leaves";
import { MlRandomForestAdapter } from "../rf";
import type { RfAdapter } from "../rf";
import type {
  ArfIterationMetrics,
  ArfModel,
  FitOptions,
  RowRecord,
  Schema,
  TableData,
} from "../types";

interface EncodedData {
  features: number[][];
  labels: number[];
  realEncoded: number[][];
}

interface EncodingState {
  encodeRows: (rows: RowRecord[]) => number[][];
  categoryCodebooks: Record<string, unknown[]>;
}

interface ArfTrainingDependencies {
  createRfAdapter: (seed: number, options: FitOptions) => RfAdapter;
}

type RNG = () => number;

interface TreeLeafAssignments {
  rowLeafByTree: string[][];
  leafMembersByTree: Array<Map<string, number[]>>;
}

interface TreeNode {
  splitColumn?: number;
  splitValue?: number;
  left?: TreeNode;
  right?: TreeNode;
}

function defaultDependencies(): ArfTrainingDependencies {
  return {
    createRfAdapter: (seed: number, options: FitOptions) =>
      new MlRandomForestAdapter({
        seed,
        nEstimators: options.numTrees,
        maxFeatures: options.maxFeatures,
        noOOB: options.useOobAccuracy === false ? true : false,
        treeOptions: {
          minNumSamples: options.minNodeSize,
          gainThreshold: options.gainThreshold,
        },
      }),
  };
}

function shuffledCopy<T>(values: T[], rng: RNG): T[] {
  const out = [...values];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = out[i];
    out[i] = out[j] as T;
    out[j] = tmp as T;
  }
  return out;
}

function generateInitialSynthetic(realRows: RowRecord[], schema: Schema, rng: RNG): RowRecord[] {
  const columns = Object.keys(schema);
  const shuffledColumns: Record<string, unknown[]> = {};

  for (const column of columns) {
    shuffledColumns[column] = shuffledCopy(realRows.map((row) => row[column]), rng);
  }

  const rows: RowRecord[] = [];
  for (let i = 0; i < realRows.length; i += 1) {
    const row: RowRecord = {};
    for (const column of columns) {
      row[column] = shuffledColumns[column]?.[i];
    }
    rows.push(row);
  }

  return rows;
}

function createEncodingState(realRows: RowRecord[], schema: Schema): EncodingState {
  const columns = Object.keys(schema);
  const categoricalMaps: Record<string, Map<unknown, number>> = {};
  const categoryCodebooks: Record<string, unknown[]> = {};

  for (const [column, kind] of Object.entries(schema)) {
    if (kind === "categorical") {
      const map = new Map<unknown, number>();
      for (const row of realRows) {
        const value = row[column];
        if (!map.has(value)) {
          map.set(value, map.size);
        }
      }
      categoricalMaps[column] = map;
      categoryCodebooks[column] = [...map.entries()]
        .sort((a, b) => a[1] - b[1])
        .map(([value]) => value);
    }
  }

  return {
    encodeRows: (rows: RowRecord[]): number[][] =>
      rows.map((row) =>
        columns.map((column) => {
          const kind = schema[column];
          const value = row[column];

          if (kind === "numeric") {
            const numeric = Number(value);
            if (Number.isNaN(numeric)) {
              throw new Error(`Column '${column}' includes non-numeric values.`);
            }
            return numeric;
          }

          const mapping = categoricalMaps[column];
          if (!mapping) {
            throw new Error(`Missing categorical mapping for '${column}'.`);
          }
          if (!mapping.has(value)) {
            mapping.set(value, mapping.size);
          }
          return mapping.get(value) ?? 0;
        }),
      ),
    categoryCodebooks,
  };
}

function buildEncodedData(
  realRows: RowRecord[],
  syntheticRows: RowRecord[],
  encoder: EncodingState,
): EncodedData {
  const realEncoded = encoder.encodeRows(realRows);
  const synthEncoded = encoder.encodeRows(syntheticRows);

  return {
    features: [...realEncoded, ...synthEncoded],
    labels: [
      ...new Array(realRows.length).fill(0),
      ...new Array(syntheticRows.length).fill(1),
    ],
    realEncoded,
  };
}

function leafPathForRow(root: TreeNode, row: number[], featureIndexMap: number[]): string {
  let current: TreeNode | undefined = root;
  let path = "";

  while (
    current &&
    typeof current.splitColumn === "number" &&
    typeof current.splitValue === "number" &&
    current.left &&
    current.right
  ) {
    const sourceColumn = featureIndexMap[current.splitColumn];
    if (sourceColumn === undefined) {
      throw new Error("Tree feature index is out of range.");
    }

    const value = row[sourceColumn];
    if (value === undefined) {
      throw new Error("Encoded row is missing a feature value.");
    }

    if (value < current.splitValue) {
      path += "L";
      current = current.left;
    } else {
      path += "R";
      current = current.right;
    }
  }

  return path || "root";
}

function getTreeLeafAssignments(adapter: RfAdapter, realEncodedRows: number[][]): TreeLeafAssignments {
  const forest = adapter.getForestSnapshot();

  const rowLeafByTree: string[][] = [];
  const leafMembersByTree: Array<Map<string, number[]>> = [];

  for (let treeIndex = 0; treeIndex < forest.trees.length; treeIndex += 1) {
    const tree = forest.trees[treeIndex] as { root?: TreeNode };
    const featureIndexes = forest.featureIndexes[treeIndex];

    if (!tree?.root || !featureIndexes) {
      throw new Error(`Invalid forest snapshot for tree ${treeIndex}.`);
    }

    const leavesForRows: string[] = [];
    const members = new Map<string, number[]>();

    for (let rowIndex = 0; rowIndex < realEncodedRows.length; rowIndex += 1) {
      const row = realEncodedRows[rowIndex];
      if (!row) {
        throw new Error(`Missing encoded row at index ${rowIndex}.`);
      }
      const path = leafPathForRow(tree.root, row, featureIndexes);
      leavesForRows.push(path);
      const indices = members.get(path) ?? [];
      indices.push(rowIndex);
      members.set(path, indices);
    }

    rowLeafByTree.push(leavesForRows);
    leafMembersByTree.push(members);
  }

  return { rowLeafByTree, leafMembersByTree };
}

function drawAdversarialSynthetic(
  realRows: RowRecord[],
  assignments: TreeLeafAssignments,
  schema: Schema,
  rng: RNG,
): RowRecord[] {
  const treeCount = assignments.rowLeafByTree.length;
  const rowCount = realRows.length;
  const columns = Object.keys(schema);

  const syntheticRows: RowRecord[] = [];

  for (let i = 0; i < rowCount; i += 1) {
    const treeIndex = Math.floor(rng() * treeCount);
    const refObs = Math.floor(rng() * rowCount);

    const leaf = assignments.rowLeafByTree[treeIndex]?.[refObs];
    if (!leaf) {
      throw new Error("Failed to resolve sampled leaf.");
    }

    const members = assignments.leafMembersByTree[treeIndex]?.get(leaf);
    if (!members || members.length === 0) {
      throw new Error("Sampled leaf has no member rows.");
    }

    const sampledRow: RowRecord = {};
    for (const column of columns) {
      const sampleIndex = members[Math.floor(rng() * members.length)] ?? members[0];
      if (sampleIndex === undefined) {
        throw new Error("Sampled row index is invalid.");
      }
      const baseRow = realRows[sampleIndex];
      if (!baseRow) {
        throw new Error("Sampled row index is invalid.");
      }
      sampledRow[column] = baseRow[column];
    }

    syntheticRows.push(sampledRow);
  }

  return syntheticRows;
}

function computeAccuracy(adapter: RfAdapter, features: number[][], labels: number[], useOob: boolean): number {
  if (useOob) {
    const oob = adapter.getOobAccuracy();
    if (oob !== null) {
      return oob;
    }
  }

  const predictions = adapter.predict(features);
  let correct = 0;
  for (let i = 0; i < labels.length; i += 1) {
    if (predictions[i] === labels[i]) {
      correct += 1;
    }
  }
  return correct / labels.length;
}

export function runArfTraining(
  input: TableData,
  options: FitOptions,
  dependencies?: Partial<ArfTrainingDependencies>,
): ArfModel {
  const seed = options.seed ?? 42;
  const delta = options.delta ?? 0;
  const maxIterations = options.maxIterations ?? 10;
  const earlyStop = options.earlyStop ?? true;
  const useOobAccuracy = options.useOobAccuracy ?? false;
  const threshold = 0.5 + delta;

  const normalizedOptions: FitOptions = {
    ...options,
    seed,
    delta,
    maxIterations,
    earlyStop,
    useOobAccuracy,
    numTrees: options.numTrees ?? 30,
    minNodeSize: options.minNodeSize ?? 5,
  };

  const deps = { ...defaultDependencies(), ...dependencies };
  const rng = seedrandom(String(seed));
  const encoder = createEncodingState(input.rows, input.schema);

  let syntheticRows = generateInitialSynthetic(input.rows, input.schema, rng);

  const history: ArfIterationMetrics[] = [];
  let iterations = 0;

  let currentEncoded = buildEncodedData(input.rows, syntheticRows, encoder);
  let currentAdapter = deps.createRfAdapter(seed, normalizedOptions);
  currentAdapter.fit({ features: currentEncoded.features, labels: currentEncoded.labels });
  let currentAcc = computeAccuracy(currentAdapter, currentEncoded.features, currentEncoded.labels, useOobAccuracy);
  history.push({ iteration: iterations, accuracy: currentAcc, threshold, converged: currentAcc <= threshold });

  while (currentAcc > threshold && iterations < maxIterations) {
    const assignments = getTreeLeafAssignments(currentAdapter, currentEncoded.realEncoded);
    syntheticRows = drawAdversarialSynthetic(input.rows, assignments, input.schema, rng);

    const nextEncoded = buildEncodedData(input.rows, syntheticRows, encoder);
    const nextAdapter = deps.createRfAdapter(seed + iterations + 1, normalizedOptions);
    nextAdapter.fit({ features: nextEncoded.features, labels: nextEncoded.labels });

    const nextAcc = computeAccuracy(nextAdapter, nextEncoded.features, nextEncoded.labels, useOobAccuracy);
    iterations += 1;

    const plateau = earlyStop && nextAcc > currentAcc;
    const converged = nextAcc <= threshold || iterations >= maxIterations || plateau;

    history.push({
      iteration: iterations,
      accuracy: nextAcc,
      threshold,
      converged: nextAcc <= threshold,
    });

    if (converged) {
      break;
    }

    currentAdapter = nextAdapter;
    currentEncoded = nextEncoded;
    currentAcc = nextAcc;
  }

  const leafModel = fitLeafParameters({
    rows: input.rows,
    encodedRows: currentEncoded.realEncoded,
    schema: input.schema,
    forest: currentAdapter.getForestSnapshot(),
    categoryCodebooks: encoder.categoryCodebooks,
    laplaceAlpha: options.laplaceAlpha ?? 0,
  });
  const leafSummary = summarizeLeafCoverage(leafModel);

  const finalHistory = history;
  const last = finalHistory[finalHistory.length - 1];
  const terminationReason = last && last.accuracy <= threshold ? "converged" : "max_iterations";

  return {
    schema: input.schema,
    fitted: true,
    metadata: {
      rowCount: input.rows.length,
      seed,
      training: {
        history: finalHistory,
        terminationReason,
        delta,
        maxIterations,
        leafModel,
        leafSummary: {
          treeCount: leafSummary.length,
          totalLeafCount: leafSummary.reduce((sum, tree) => sum + tree.leafCount, 0),
          coverageSums: leafSummary.map((tree) => tree.coverageSum),
        },
      },
    },
  };
}
