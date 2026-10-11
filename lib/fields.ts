// Field metadata shared by the client and the API (no model weights here).

export type FeatureKey = "N" | "P" | "K" | "temperature" | "humidity" | "ph" | "rainfall";
export type CropInput = Record<FeatureKey, number>;

export const FIELDS: {
  key: FeatureKey;
  label: string;
  hint: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  group: "Soil" | "Climate";
}[] = [
  { key: "N", label: "Nitrogen", hint: "N", unit: "kg/ha", min: 0, max: 140, step: 1, group: "Soil" },
  { key: "P", label: "Phosphorus", hint: "P", unit: "kg/ha", min: 5, max: 145, step: 1, group: "Soil" },
  { key: "K", label: "Potassium", hint: "K", unit: "kg/ha", min: 5, max: 205, step: 1, group: "Soil" },
  { key: "ph", label: "Soil pH", hint: "acidity", unit: "pH", min: 3.5, max: 10, step: 0.1, group: "Soil" },
  { key: "temperature", label: "Temperature", hint: "average", unit: "°C", min: 8, max: 44, step: 0.5, group: "Climate" },
  { key: "humidity", label: "Humidity", hint: "relative", unit: "%", min: 14, max: 100, step: 1, group: "Climate" },
  { key: "rainfall", label: "Rainfall", hint: "seasonal", unit: "mm", min: 20, max: 300, step: 1, group: "Climate" },
];

export const CROPS: Record<string, { name: string; kind: string }> = {
  rice: { name: "Rice", kind: "Cereal" },
  maize: { name: "Maize", kind: "Cereal" },
  chickpea: { name: "Chickpea", kind: "Pulse" },
  kidneybeans: { name: "Kidney beans", kind: "Pulse" },
  pigeonpeas: { name: "Pigeon peas", kind: "Pulse" },
  mothbeans: { name: "Moth beans", kind: "Pulse" },
  mungbean: { name: "Mung bean", kind: "Pulse" },
  blackgram: { name: "Black gram", kind: "Pulse" },
  lentil: { name: "Lentil", kind: "Pulse" },
  pomegranate: { name: "Pomegranate", kind: "Fruit" },
  banana: { name: "Banana", kind: "Fruit" },
  mango: { name: "Mango", kind: "Fruit" },
  grapes: { name: "Grapes", kind: "Fruit" },
  watermelon: { name: "Watermelon", kind: "Fruit" },
  muskmelon: { name: "Muskmelon", kind: "Fruit" },
  apple: { name: "Apple", kind: "Fruit" },
  orange: { name: "Orange", kind: "Fruit" },
  papaya: { name: "Papaya", kind: "Fruit" },
  coconut: { name: "Coconut", kind: "Plantation" },
  coffee: { name: "Coffee", kind: "Plantation" },
  cotton: { name: "Cotton", kind: "Fibre" },
  jute: { name: "Jute", kind: "Fibre" },
};

export type Band = { p10: number; p50: number; p90: number };

export type Ranked = {
  id: string;
  name: string;
  kind: string;
  probability: number;
  profile: Record<FeatureKey, Band>;
};
