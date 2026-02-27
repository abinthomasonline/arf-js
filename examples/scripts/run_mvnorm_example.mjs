import fs from "node:fs";
import { fit, sample } from "../../dist/index.js";

const rows = JSON.parse(fs.readFileSync("examples/data/mvnorm_train.json", "utf8"));
const schema = {
  var1: "numeric",
  var2: "numeric",
};

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
    missingPolicy: "reject",
  },
);

const synthetic = sample(model, 2000, 2023);
fs.writeFileSync("examples/output/mvnorm_synth.json", JSON.stringify(synthetic, null, 2));
fs.writeFileSync("examples/output/mvnorm_training.json", JSON.stringify(model.metadata.training, null, 2));

console.log("termination=", model.metadata.training.terminationReason);
console.log("history=", model.metadata.training.history);
console.log("wrote examples/output/mvnorm_synth.json");
