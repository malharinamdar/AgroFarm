import type { Band } from "@/lib/fields";

type Props = { label: string; unit: string; min: number; max: number; band: Band; value: number };

export default function RangeStrip({ label, unit, min, max, band, value }: Props) {
  const x = (v: number) => `${(((Math.min(max, Math.max(min, v)) - min) / (max - min)) * 100).toFixed(2)}%`;
  const status = value < band.p10 ? "low" : value > band.p90 ? "high" : null;
  const fmt = (v: number) => (Math.abs(v) < 20 ? v.toFixed(1) : Math.round(v).toString());

  return (
    <div className="grid grid-cols-[96px_1fr_64px] items-center gap-4 py-2.5 sm:grid-cols-[120px_1fr_76px]">
      <span className="text-[13px]">
        {label}
        {status && <span className="ml-1.5 text-[11px] text-clay">{status}</span>}
      </span>
      <div className="relative h-5" title={`${fmt(band.p10)}–${fmt(band.p90)} ${unit}`}>
        <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
        <div
          className="absolute top-1/2 h-2 -translate-y-1/2 rounded-full bg-leaf-soft transition-all duration-500"
          style={{ left: x(band.p10), width: `calc(${x(band.p90)} - ${x(band.p10)})` }}
        />
        <div
          className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-leaf transition-all duration-500"
          style={{ left: x(band.p50) }}
        />
        <div
          className={`absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-card transition-all duration-300 ${
            status ? "bg-clay" : "bg-ink"
          }`}
          style={{ left: x(value) }}
        />
      </div>
      <span className="num text-right text-xs text-muted">
        {fmt(value)}
        <span className="text-faint"> {unit === "pH" ? "" : unit}</span>
      </span>
    </div>
  );
}
