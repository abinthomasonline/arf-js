import type { ForestSnapshot } from "../rf";
import type {
  ColumnKind,
  LeafFeatureParams,
  LeafFitModel,
  LeafParameters,
  RowRecord,
  Schema,
} from "../types";

interface TreeNode {
  splitColumn?: number;
  splitValue?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface LeafFitInput {
  rows: RowRecord[];
  encodedRows: number[][];
  schema: Schema;
  forest: ForestSnapshot;
  categoryCodebooks?: Record<string, unknown[]>;
  laplaceAlpha?: number;
}

function isNumericColumn(kind: ColumnKind): kind is "numeric" {
  return kind === "numeric";
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

interface FeatureBounds {
  min: number;
  max: number;
}

function createInitialBounds(featureCount: number): FeatureBounds[] {
  return new Array(featureCount).fill(null).map(() => ({ min: -Infinity, max: Infinity }));
}

function enumerateLeafBounds(
  node: TreeNode,
  featureIndexMap: number[],
  featureCount: number,
  path: string,
  bounds: FeatureBounds[],
  out: Map<string, FeatureBounds[]>,
): void {
  if (
    typeof node.splitColumn === "number" &&
    typeof node.splitValue === "number" &&
    node.left &&
    node.right
  ) {
    const sourceColumn = featureIndexMap[node.splitColumn];
    if (sourceColumn === undefined) {
      throw new Error("Tree feature index is out of range.");
    }

    const leftBounds = bounds.map((item) => ({ ...item }));
    leftBounds[sourceColumn] = {
      min: leftBounds[sourceColumn]?.min ?? -Infinity,
      max: Math.min(leftBounds[sourceColumn]?.max ?? Infinity, node.splitValue),
    };

    const rightBounds = bounds.map((item) => ({ ...item }));
    rightBounds[sourceColumn] = {
      min: Math.max(rightBounds[sourceColumn]?.min ?? -Infinity, node.splitValue),
      max: rightBounds[sourceColumn]?.max ?? Infinity,
    };

    enumerateLeafBounds(node.left, featureIndexMap, featureCount, `${path}L`, leftBounds, out);
    enumerateLeafBounds(node.right, featureIndexMap, featureCount, `${path}R`, rightBounds, out);
    return;
  }

  out.set(path || "root", bounds);
}

function fitNumeric(column: string, rows: RowRecord[], bounds: FeatureBounds): LeafFeatureParams {
  const numericValues = rows.map((row) => Number(row[column]));
  if (numericValues.some((value) => Number.isNaN(value))) {
    throw new Error(`Column '${column}' includes non-numeric values.`);
  }

  const mean = numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length;
  const variance =
    numericValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    Math.max(numericValues.length, 1);

  return {
    kind: "numeric",
    min: bounds.min,
    max: bounds.max,
    mean,
    std: Math.sqrt(variance),
  };
}

function fitCategorical(
  column: string,
  rows: RowRecord[],
  codebook: unknown[] | undefined,
  bounds: FeatureBounds,
  laplaceAlpha: number,
): LeafFeatureParams {
  const counts = new Map<unknown, number>();
  for (const row of rows) {
    const value = row[column];
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }

  if (!codebook || laplaceAlpha <= 0) {
    const entries = [...counts.entries()];
    return {
      kind: "categorical",
      values: entries.map(([value]) => value),
      probabilities: entries.map(([, count]) => count / rows.length),
    };
  }

  const valid: unknown[] = [];
  for (let code = 0; code < codebook.length; code += 1) {
    if (code >= bounds.min && code <= bounds.max) {
      const value = codebook[code];
      if (value !== undefined) {
        valid.push(value);
      }
    }
  }

  const values = valid.length > 0 ? valid : codebook;
  const k = values.length;
  const total = rows.length;

  const probabilities = values.map((value) => ((counts.get(value) ?? 0) + laplaceAlpha) / (total + laplaceAlpha * k));

  return {
    kind: "categorical",
    values,
    probabilities,
  };
}

function fitLeafFeatureParams(
  schema: Schema,
  rows: RowRecord[],
  boundsByColumn: Record<string, FeatureBounds>,
  categoryCodebooks: Record<string, unknown[]>,
  laplaceAlpha: number,
): Record<string, LeafFeatureParams> {
  const params: Record<string, LeafFeatureParams> = {};

  for (const [column, kind] of Object.entries(schema)) {
    const bounds = boundsByColumn[column] ?? { min: -Infinity, max: Infinity };
    params[column] = isNumericColumn(kind)
      ? fitNumeric(column, rows, bounds)
      : fitCategorical(column, rows, categoryCodebooks[column], bounds, laplaceAlpha);
  }

  return params;
}

export function fitLeafParameters(input: LeafFitInput): LeafFitModel {
  if (input.rows.length !== input.encodedRows.length) {
    throw new Error("rows and encodedRows length must match.");
  }

  const columns = Object.keys(input.schema);
  const laplaceAlpha = input.laplaceAlpha ?? 0;
  const categoryCodebooks = input.categoryCodebooks ?? {};

  const trees = input.forest.trees.map((tree, treeIndex) => {
    const featureIndexes = input.forest.featureIndexes[treeIndex];
    if (!featureIndexes) {
      throw new Error(`Missing feature indexes for tree ${treeIndex}.`);
    }

    const leafGroups = new Map<string, number[]>();
    const root = (tree as { root?: TreeNode }).root;
    if (!root) {
      throw new Error(`Tree ${treeIndex} has no root node.`);
    }

    const leafBoundsByPath = new Map<string, FeatureBounds[]>();
    enumerateLeafBounds(
      root,
      featureIndexes,
      columns.length,
      "",
      createInitialBounds(columns.length),
      leafBoundsByPath,
    );

    for (let rowIndex = 0; rowIndex < input.encodedRows.length; rowIndex += 1) {
      const encodedRow = input.encodedRows[rowIndex];
      if (!encodedRow) {
        throw new Error(`Missing encoded row at index ${rowIndex}.`);
      }

      const path = leafPathForRow(root, encodedRow, featureIndexes);
      const members = leafGroups.get(path) ?? [];
      members.push(rowIndex);
      leafGroups.set(path, members);
    }

    const leaves: LeafParameters[] = [...leafGroups.entries()].map(([path, rowIndexes]) => {
      const leafRows = rowIndexes.map((index) => input.rows[index]).filter(Boolean) as RowRecord[];
      const featureBounds = leafBoundsByPath.get(path) ?? createInitialBounds(columns.length);
      const boundsByColumn: Record<string, FeatureBounds> = {};
      for (let i = 0; i < columns.length; i += 1) {
        const column = columns[i];
        if (column) {
          boundsByColumn[column] = featureBounds[i] ?? { min: -Infinity, max: Infinity };
        }
      }
      return {
        path,
        count: rowIndexes.length,
        coverage: rowIndexes.length / input.rows.length,
        features: fitLeafFeatureParams(
          input.schema,
          leafRows,
          boundsByColumn,
          categoryCodebooks,
          laplaceAlpha,
        ),
      };
    });

    return {
      treeIndex,
      leaves,
    };
  });

  return { trees };
}

export interface LeafCoverageSummary {
  treeIndex: number;
  coverageSum: number;
  leafCount: number;
}

export function summarizeLeafCoverage(model: LeafFitModel): LeafCoverageSummary[] {
  return model.trees.map((tree) => ({
    treeIndex: tree.treeIndex,
    coverageSum: tree.leaves.reduce((sum, leaf) => sum + leaf.coverage, 0),
    leafCount: tree.leaves.length,
  }));
}
