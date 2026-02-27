import fs from "node:fs";
import { fit, sample } from "../../dist/index.js";

const rows = JSON.parse(fs.readFileSync("examples/data/digits_zero_train.json", "utf8"));
const schema = Object.fromEntries(Object.keys(rows[0] ?? {}).map((column) => [column, "categorical"]));

const model = fit(
  { rows, schema },
  {
    seed: 2023,
    delta: 0,
    maxIterations: 10,
    numTrees: 30,
    minNodeSize: 5,
    gainThreshold: 0,
    earlyStop: true,
    useOobAccuracy: true,
    laplaceAlpha: 1,
    missingPolicy: "reject",
  },
);

const synthetic = sample(model, 4, 2023);
fs.writeFileSync("examples/output/digits_synth.json", JSON.stringify(synthetic, null, 2));
fs.writeFileSync("examples/output/digits_training.json", JSON.stringify(model.metadata.training, null, 2));

console.log("termination=", model.metadata.training.terminationReason);
console.log("history=", model.metadata.training.history);
console.log("wrote examples/output/digits_synth.json");
