"use client";

import { useState } from "react";

type Props = {
  id: string;
  label: string;
  hint?: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
};

export default function NumberField({ id, label, hint, unit, min, max, step, value, onChange }: Props) {
  const fill = `${(((value - min) / (max - min)) * 100).toFixed(1)}%`;
  const decimals = step < 1 ? 1 : 0;
  // Free typing; clamp only on blur so partial values like "1" (on the way to "165") aren't snapped.
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div className="py-3.5">
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-[15px]">
          {label}
          {hint && <span className="ml-2 text-xs text-faint">{hint}</span>}
        </label>
        <div className="flex items-baseline gap-1.5">
          <input
            type="number"
            inputMode="decimal"
            aria-label={`${label} value`}
            className="num w-16 rounded-md border border-transparent bg-transparent px-1.5 py-0.5 text-right text-[15px] outline-none transition-colors hover:border-line focus:border-leaf focus:bg-card"
            min={min}
            max={max}
            step={step}
            value={draft ?? Number(value.toFixed(decimals))}
            onChange={(e) => {
              setDraft(e.target.value);
              const v = Number(e.target.value);
              if (e.target.value !== "" && v >= min && v <= max) onChange(v);
            }}
            onBlur={() => {
              const v = Number(draft);
              if (draft && Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)));
              setDraft(null);
            }}
          />
          <span className="num w-10 text-xs text-faint">{unit}</span>
        </div>
      </div>
      <input
        id={id}
        type="range"
        className="range mt-1.5"
        style={{ ["--fill" as string]: fill }}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="num mt-0.5 flex justify-between text-[10.5px] text-faint">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
