import raw from "./model/crop.json";
import { predictProba, type Tree } from "./forest";
import { CROPS, FIELDS, type Band, type CropInput, type FeatureKey, type Ranked } from "./fields";

export { FIELDS };

type CropModel = {
  features: FeatureKey[];
  classes: string[];
  trees: Tree[];
  profiles: Record<string, Record<FeatureKey, Band>>;
  ranges: Record<FeatureKey, [number, number]>;
};
const model = raw as unknown as CropModel;

export const CROP_IDS = model.classes;

/** Median growing conditions for a crop in the training data, used for the sample presets. */
export function medianConditions(crop: string): CropInput {
  const p = model.profiles[crop];
  return Object.fromEntries(model.features.map((f) => [f, p[f].p50])) as CropInput;
}

export function parseCropInput(body: unknown): { value: CropInput } | { error: string } {
  if (!body || typeof body !== "object") return { error: "Expected a JSON object." };
  const value = {} as CropInput;
  for (const f of FIELDS) {
    const v = Number((body as Record<string, unknown>)[f.key]);
    if (!Number.isFinite(v)) return { error: `${f.label} must be a number.` };
    if (v < f.min || v > f.max) return { error: `${f.label} must be between ${f.min} and ${f.max} ${f.unit}.` };
    value[f.key] = v;
  }
  return { value };
}


export function recommendCrops(input: CropInput): { ranking: Ranked[] } {
  const x = model.features.map((f) => input[f]);
  const proba = predictProba(model.trees, x, model.classes.length);
  const ranking = model.classes
    .map((id, i) => ({ id, p: proba[i] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5)
    .map(({ id, p }) => ({
      id,
      name: CROPS[id]?.name ?? id,
      kind: CROPS[id]?.kind ?? "",
      probability: Math.round(p * 1000) / 1000,
      profile: model.profiles[id],
    }));
  return { ranking };
}
