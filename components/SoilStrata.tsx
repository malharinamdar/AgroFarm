// Decorative soil profile above the footer: grass, topsoil, subsoil, parent rock.

const LAYERS = [
  { h: "h-7", bg: "#5d4a37", dots: "rgba(255,255,255,0.10)", size: "7px 7px", label: "A · topsoil" },
  { h: "h-6", bg: "#8c6d4c", dots: "rgba(255,255,255,0.12)", size: "11px 9px", label: "B · subsoil" },
  { h: "h-5", bg: "#bba483", dots: "rgba(23,36,27,0.14)", size: "17px 13px", label: "C · parent rock" },
];

export default function SoilStrata() {
  return (
    <div aria-hidden className="mt-24">
      <svg className="block h-4 w-full" preserveAspectRatio="none" viewBox="0 0 1200 16">
        <defs>
          <pattern id="grass" width="24" height="16" patternUnits="userSpaceOnUse">
            <path d="M3 16 L5 5 M7 16 L8 2 M11 16 L14 7 M16 16 L15 4 M20 16 L22 8" stroke="#3f7a52" strokeWidth="1.3" fill="none" />
          </pattern>
        </defs>
        <rect width="1200" height="16" fill="url(#grass)" />
      </svg>
      {LAYERS.map((l) => (
        <div
          key={l.label}
          className={`${l.h} relative`}
          style={{
            backgroundColor: l.bg,
            backgroundImage: `radial-gradient(${l.dots} 1.2px, transparent 1.6px)`,
            backgroundSize: l.size,
          }}
        >
          <span className="num absolute right-5 top-1/2 -translate-y-1/2 text-[9.5px] uppercase tracking-[0.16em] text-white/55 sm:right-8">
            {l.label}
          </span>
        </div>
      ))}
    </div>
  );
}
