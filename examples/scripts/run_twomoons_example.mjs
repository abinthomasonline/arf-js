import fs from "node:fs";
import { fit, sample } from "../../dist/index.js";

const rows = JSON.parse(fs.readFileSync("examples/data/twomoons_train.json", "utf8"));
const schema = {
  dim_1: "numeric",
  dim_2: "numeric",
  target: "categorical",
};

const model = fit(
  { rows, schema },
  {
    seed: 2022,
    delta: 0,
    maxIterations: 12,
    numTrees: 20,
    minNodeSize: 2,
    maxFeatures: 2,
    gainThreshold: 0,
    earlyStop: false,
    useOobAccuracy: true,
    laplaceAlpha: 1,
    missingPolicy: "reject",
  },
);

const synthetic = sample(model, 1000, 2022);
fs.writeFileSync("examples/output/twomoons_synth.json", JSON.stringify(synthetic, null, 2));
fs.writeFileSync("examples/output/twomoons_training.json", JSON.stringify(model.metadata.training, null, 2));

console.log("termination=", model.metadata.training.terminationReason);
console.log("iters=", model.metadata.training.history.length - 1);
console.log("last=", model.metadata.training.history[model.metadata.training.history.length - 1]);
console.log("wrote examples/output/twomoons_synth.json");
