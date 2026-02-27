import fs from "node:fs";
import { fit, sample } from "../../dist/index.js";

const rows = JSON.parse(fs.readFileSync("examples/data/titanic_train.json", "utf8"));

const schema = {
  pclass: "categorical",
  sex: "categorical",
  age: "numeric",
  sibsp: "categorical",
  parch: "categorical",
  fare: "numeric",
  embarked: "categorical",
  class: "categorical",
  adult_male: "categorical",
  alone: "categorical",
  survived: "categorical",
};

const model = fit(
  { rows, schema },
  {
    seed: 2026,
    delta: 0,
    maxIterations: 10,
    numTrees: 30,
    minNodeSize: 5,
    gainThreshold: 0,
    earlyStop: true,
    useOobAccuracy: true,
    laplaceAlpha: 1,
    missingPolicy: "impute",
  },
);

const synthetic = sample(model, rows.length, 2026);

fs.writeFileSync("examples/output/titanic_synth.json", JSON.stringify(synthetic, null, 2));
fs.writeFileSync("examples/output/titanic_training.json", JSON.stringify(model.metadata.training, null, 2));

console.log("termination=", model.metadata.training.terminationReason);
console.log("history=", model.metadata.training.history);
console.log("wrote examples/output/titanic_synth.json");
