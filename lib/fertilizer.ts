import raw from "./model/fertilizer.json";
import { predictProba, type Tree } from "./forest";

type FertModel = {
  numeric: NumericKey[];
  soils: string[];
  crops: string[];
  features: string[];
  classes: string[];
  tree: Tree;
  ranges: Record<NumericKey, [number, number]>;
};
const model = raw as unknown as FertModel;

export type NumericKey = "Temperature" | "Humidity" | "Moisture" | "Nitrogen" | "Potassium" | "Phosphorous";
export type FertInput = Record<NumericKey, number> & { soil: string; crop: string };

export const SOILS = model.soils;
export const FERT_CROPS = model.crops;

export const FERT_FIELDS: { key: NumericKey; label: string; unit: string; min: number; max: number }[] = [
  { key: "Nitrogen", label: "Nitrogen", unit: "kg/ha", min: 0, max: 50 },
  { key: "Phosphorous", label: "Phosphorus", unit: "kg/ha", min: 0, max: 50 },
  { key: "Potassium", label: "Potassium", unit: "kg/ha", min: 0, max: 25 },
  { key: "Temperature", label: "Temperature", unit: "°C", min: 20, max: 45 },
  { key: "Humidity", label: "Humidity", unit: "%", min: 40, max: 80 },
  { key: "Moisture", label: "Soil moisture", unit: "%", min: 20, max: 70 },
];

/** Grade = N–P₂O₅–K₂O percentages by weight. */
export const FERTILIZERS: Record<string, { grade: [number, number, number]; note: string }> = {
  Urea: { grade: [46, 0, 0], note: "Straight nitrogen. Corrects nitrogen deficit when P and K are adequate." },
  DAP: { grade: [18, 46, 0], note: "Diammonium phosphate. Phosphorus-heavy with a nitrogen top-up." },
  "14-35-14": { grade: [14, 35, 14], note: "Complex NPK, phosphorus-led, with balanced N and K." },
  "28-28": { grade: [28, 28, 0], note: "Equal nitrogen and phosphorus, no potash." },
  "17-17-17": { grade: [17, 17, 17], note: "Balanced NPK for fields short on all three." },
  "20-20": { grade: [20, 20, 0], note: "Nitrogen and phosphorus in equal parts, no potash." },
  "10-26-26": { grade: [10, 26, 26], note: "Low nitrogen, high phosphorus and potash." },
};

export function parseFertInput(body: unknown): { value: FertInput } | { error: string } {
  if (!body || typeof body !== "object") return { error: "Expected a JSON object." };
  const b = body as Record<string, unknown>;
  if (typeof b.soil !== "string" || !SOILS.includes(b.soil)) return { error: "Unknown soil type." };
  if (typeof b.crop !== "string" || !FERT_CROPS.includes(b.crop)) return { error: "Unknown crop type." };
  const value = { soil: b.soil, crop: b.crop } as FertInput;
  for (const f of FERT_FIELDS) {
    const v = Number(b[f.key]);
    if (!Number.isFinite(v) || v < f.min || v > f.max)
      return { error: `${f.label} must be between ${f.min} and ${f.max} ${f.unit}.` };
    value[f.key] = v;
  }
  return { value };
}

export function recommendFertilizer(input: FertInput) {
  const x = model.features.map((name) => {
    if (name.startsWith("soil=")) return name.slice(5) === input.soil ? 1 : 0;
    if (name.startsWith("crop=")) return name.slice(5) === input.crop ? 1 : 0;
    return input[name as NumericKey];
  });
  const proba = predictProba([model.tree], x, model.classes.length);
  const best = proba.indexOf(Math.max(...proba));
  const name = model.classes[best];
  return { name, ...FERTILIZERS[name] };
}
