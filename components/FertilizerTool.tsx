"use client";

import { useState } from "react";
import NumberField from "./NumberField";
import { FERT_FIELDS, type NumericKey } from "@/lib/fertilizer";

type Result = { name: string; grade: [number, number, number]; note: string };

const START: Record<NumericKey, number> = {
  Nitrogen: 12,
  Phosphorous: 20,
  Potassium: 6,
  Temperature: 30,
  Humidity: 60,
  Moisture: 45,
};

export default function FertilizerTool({ soils, crops }: { soils: string[]; crops: string[] }) {
  const [soil, setSoil] = useState(soils.includes("Loamy") ? "Loamy" : soils[0]);
  const [crop, setCrop] = useState(crops.includes("Wheat") ? "Wheat" : crops[0]);
  const [values, setValues] = useState(START);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/fertilizer", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...values, soil, crop }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Request failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-10 lg:grid-cols-[minmax(0,6fr)_minmax(0,6fr)] lg:gap-14">
      <form onSubmit={submit}>
        <Choice label="Soil type" options={soils} value={soil} onChange={setSoil} />
        <Choice label="Crop" options={crops} value={crop} onChange={setCrop} />

        <fieldset className="mt-8">
          <legend className="num mb-1 w-full border-b border-line pb-2 text-[11px] uppercase tracking-[0.16em] text-muted">
            Soil test &amp; weather
          </legend>
          <div className="grid gap-x-8 divide-y divide-line/70 sm:grid-cols-2 sm:divide-y-0">
            {FERT_FIELDS.map((f) => (
              <NumberField
                key={f.key}
                id={`fx-${f.key}`}
                label={f.label}
                unit={f.unit}
                min={f.min}
                max={f.max}
                step={1}
                value={values[f.key]}
                onChange={(v) => setValues((s) => ({ ...s, [f.key]: v }))}
              />
            ))}
          </div>
        </fieldset>

        <button
          type="submit"
          disabled={loading}
          className="mt-8 rounded-full bg-ink px-6 py-3 text-[15px] text-paper transition-transform active:scale-[0.98] disabled:opacity-60"
        >
          {loading ? "Matching…" : "Recommend fertilizer"}
        </button>
        {error && <p className="mt-4 text-sm text-clay">{error}</p>}
      </form>

      <aside className="lg:sticky lg:top-6 lg:self-start">
        <div className="rounded-2xl border border-line bg-card p-6 sm:p-8">
          {result ? (
            <div key={`${result.name}-${soil}-${crop}`} className="rise">
              <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">
                Recommended for {crop.toLowerCase()} on {soil.toLowerCase()} soil
              </p>
              <h2 className="mt-3 font-display text-[56px] leading-none tracking-[-0.02em] sm:text-[72px]">
                {result.name}
              </h2>
              <p className="mt-4 max-w-sm text-[15px] leading-relaxed text-muted">{result.note}</p>
              <Grade grade={result.grade} />
            </div>
          ) : (
            <div>
              <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Recommendation</p>
              <p className="mt-3 max-w-sm font-display text-2xl leading-snug">
                The matching grade and its nutrient split will show here.
              </p>
              <Grade grade={[0, 0, 0]} muted />
            </div>
          )}
        </div>
      </aside>
    </section>
  );
}

function Choice({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <fieldset className="mt-6 first:mt-0">
      <legend className="num mb-3 text-[11px] uppercase tracking-[0.16em] text-muted">{label}</legend>
      <div className="flex flex-wrap gap-2">
        {options.map((o) => (
          <button
            key={o}
            type="button"
            aria-pressed={o === value}
            onClick={() => onChange(o)}
            className={`rounded-full border px-3.5 py-1.5 text-[14px] transition-colors ${
              o === value ? "border-ink bg-ink text-paper" : "border-line text-muted hover:border-ink hover:text-ink"
            }`}
          >
            {o}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function Grade({ grade, muted }: { grade: [number, number, number]; muted?: boolean }) {
  const parts = [
    { k: "N", name: "Nitrogen", v: grade[0], color: "bg-leaf" },
    { k: "P₂O₅", name: "Phosphate", v: grade[1], color: "bg-ochre" },
    { k: "K₂O", name: "Potash", v: grade[2], color: "bg-clay" },
  ];
  return (
    <div className="mt-8 border-t border-line pt-6">
      <p className="text-[13px] text-faint">Nutrient content, % by weight</p>
      <div className="mt-4 grid grid-cols-3 gap-4">
        {parts.map((p) => (
          <div key={p.k}>
            <div className="flex h-28 items-end rounded-md bg-paper-2/60">
              <div
                className={`w-full rounded-md ${muted ? "bg-line" : p.color} transition-[height] duration-700`}
                style={{ height: `${Math.max(2, (p.v / 46) * 100)}%` }}
              />
            </div>
            <p className="num mt-2 text-lg">{muted ? "—" : p.v}</p>
            <p className="text-xs text-muted">
              {p.name} <span className="text-faint">{p.k}</span>
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
