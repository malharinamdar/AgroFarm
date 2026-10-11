"use client";

import { useEffect, useRef, useState } from "react";
import NumberField from "./NumberField";
import RangeStrip from "./RangeStrip";
import { FIELDS, type CropInput, type Ranked } from "@/lib/fields";

type Preset = { label: string; values: CropInput };

const START: CropInput = { N: 60, P: 45, K: 40, ph: 6.5, temperature: 25, humidity: 70, rainfall: 110 };

export default function CropTool({ presets, allCrops }: { presets: Preset[]; allCrops: string[] }) {
  const [values, setValues] = useState<CropInput>(START);
  const [ranking, setRanking] = useState<Ranked[] | null>(null);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasRun = useRef(false);
  const request = useRef(0);

  async function run(input: CropInput) {
    const id = ++request.current;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/crop", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(input),
      });
      const data = await res.json();
      if (id !== request.current) return;
      if (!res.ok) throw new Error(data.error ?? "Request failed");
      setRanking(data.ranking);
      setSelected(0);
      hasRun.current = true;
    } catch (e) {
      if (id === request.current) setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      if (id === request.current) setLoading(false);
    }
  }

  // After the first ranking, keep it in sync with the inputs.
  useEffect(() => {
    if (!hasRun.current) return;
    const t = setTimeout(() => run(values), 200);
    return () => clearTimeout(t);
  }, [values]);

  const set = (key: keyof CropInput) => (v: number) => setValues((s) => ({ ...s, [key]: v }));

  return (
    <section className="grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-14">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(values);
        }}
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs text-faint">Load a sample</span>
          {presets.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => {
                setValues(p.values);
                if (!hasRun.current) run(p.values);
              }}
              className="rounded-full border border-line px-3 py-1 text-[13px] text-muted transition-colors hover:border-ink hover:text-ink"
            >
              {p.label}
            </button>
          ))}
        </div>

        {(["Soil", "Climate"] as const).map((group) => (
          <fieldset key={group} className="mt-8">
            <legend className="num mb-1 w-full border-b border-line pb-2 text-[11px] uppercase tracking-[0.16em] text-muted">
              {group}
            </legend>
            <div className="divide-y divide-line/70">
              {FIELDS.filter((f) => f.group === group).map((f) => (
                <NumberField
                  key={f.key}
                  id={`f-${f.key}`}
                  label={f.label}
                  hint={f.hint}
                  unit={f.unit}
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={values[f.key]}
                  onChange={set(f.key)}
                />
              ))}
            </div>
          </fieldset>
        ))}

        <div className="mt-8 flex items-center gap-4">
          <button
            type="submit"
            className="rounded-full bg-ink px-6 py-3 text-[15px] text-paper transition-transform active:scale-[0.98] disabled:opacity-60"
            disabled={loading && !ranking}
          >
            {ranking ? "Re-rank" : "Rank crops"}
          </button>
          <p className="text-[13px] text-faint">{ranking ? "Updates as you adjust." : "Takes under a second."}</p>
        </div>
        {error && <p className="mt-4 text-sm text-clay">{error}</p>}
      </form>

      <aside className="lg:sticky lg:top-6 lg:self-start">
        <div
          className={`rounded-2xl border border-line bg-card p-6 transition-opacity sm:p-8 ${
            loading && ranking ? "opacity-70" : ""
          }`}
        >
          {ranking ? (
            <Result ranking={ranking} selected={selected} onSelect={setSelected} values={values} />
          ) : (
            <Empty crops={allCrops} />
          )}
        </div>
      </aside>
    </section>
  );
}

function Empty({ crops }: { crops: string[] }) {
  return (
    <div>
      <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Ranking</p>
      <p className="mt-3 max-w-sm font-display text-2xl leading-snug">
        Set your field&apos;s conditions, or load a sample, to rank the crops.
      </p>
      <ul className="mt-8 grid grid-cols-2 gap-x-6 gap-y-1.5 text-[14px] text-faint sm:grid-cols-3">
        {crops.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
    </div>
  );
}

function Result({
  ranking,
  selected,
  onSelect,
  values,
}: {
  ranking: Ranked[];
  selected: number;
  onSelect: (i: number) => void;
  values: CropInput;
}) {
  const top = ranking[0];
  const focus = ranking[selected];
  const inside = FIELDS.filter((f) => {
    const b = focus.profile[f.key];
    return values[f.key] >= b.p10 && values[f.key] <= b.p90;
  }).length;

  return (
    <div key={top.id} className="rise">
      <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Best match</p>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
        <h2 className="font-display text-[48px] leading-none tracking-[-0.02em] sm:text-[60px]">{top.name}</h2>
        <p className="pb-1.5 text-sm text-muted">
          {top.kind} <span className="text-faint">·</span>{" "}
          <span className="num text-ink">{Math.round(top.probability * 100)}%</span> of the forest&apos;s vote
        </p>
      </div>

      <ol className="mt-7 space-y-1">
        {ranking.map((r, i) => (
          <li key={r.id}>
            <button
              type="button"
              onClick={() => onSelect(i)}
              aria-pressed={i === selected}
              className={`grid w-full grid-cols-[28px_1fr_minmax(80px,40%)_44px] items-center gap-3 rounded-lg px-2 py-1.5 text-left text-[14px] transition-colors ${
                i === selected ? "bg-paper-2/70" : "hover:bg-paper-2/40"
              }`}
            >
              <span className="num text-xs text-faint">{String(i + 1).padStart(2, "0")}</span>
              <span className={r.probability < 0.02 ? "text-faint" : ""}>{r.name}</span>
              <span className="h-[3px] rounded-full bg-line/60">
                <span
                  className="block h-full rounded-full bg-leaf transition-[width] duration-500"
                  style={{ width: `${Math.max(1, r.probability * 100)}%` }}
                />
              </span>
              <span className="num text-right text-xs text-muted">{(r.probability * 100).toFixed(0)}%</span>
            </button>
          </li>
        ))}
      </ol>

      <div className="mt-8 border-t border-line pt-6">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-display text-xl">
            Your field vs. where <em>{focus.name.toLowerCase()}</em> grows
          </h3>
          <p className="num text-xs text-muted">
            {inside}/{FIELDS.length} within range
          </p>
        </div>
        <p className="mt-1 text-[13px] text-faint">
          Band covers the middle 80% of recorded {focus.name.toLowerCase()} fields; the tick is the median.
        </p>
        <div className="mt-4 divide-y divide-line/60">
          {FIELDS.map((f) => (
            <RangeStrip
              key={f.key}
              label={f.label}
              unit={f.unit}
              min={f.min}
              max={f.max}
              band={focus.profile[f.key]}
              value={values[f.key]}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
