import fs from "node:fs";
import { fit, sample } from "../../dist/index.js";

const rows = JSON.parse(fs.readFileSync("examples/data/penguins_train.json", "utf8"));

const schema = {
  species: "categorical",
  island: "categorical",
  bill_length_mm: "numeric",
  bill_depth_mm: "numeric",
  flipper_length_mm: "numeric",
  body_mass_g: "numeric",
  sex: "categorical",
  island_code: "categorical",
  heavy: "categorical",
};

const model = fit(
  { rows, schema },
  {
    seed: 2027,
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

const synthetic = sample(model, rows.length, 2027);

fs.writeFileSync("examples/output/penguins_synth.json", JSON.stringify(synthetic, null, 2));
fs.writeFileSync("examples/output/penguins_training.json", JSON.stringify(model.metadata.training, null, 2));

console.log("termination=", model.metadata.training.terminationReason);
console.log("history=", model.metadata.training.history);
console.log("wrote examples/output/penguins_synth.json");
