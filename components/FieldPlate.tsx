// Hero illustration: a ploughed field running to the horizon, drawn like a book plate.

const W = 480;
const H = 360;
const HORIZON = 178;
const VX = 236;

function stalk(x: number, h: number, lean: number) {
  const top = { x: x + lean, y: H - h };
  const ears = Array.from({ length: 7 }, (_, i) => {
    const t = i / 7;
    const cx = top.x - lean * 0.08 * i;
    const cy = top.y + 6 + i * 6.2;
    return (
      <g key={i}>
        <ellipse cx={cx - 3.2} cy={cy} rx="2.6" ry="5.8" transform={`rotate(-28 ${cx - 3.2} ${cy})`} opacity={1 - t * 0.2} />
        <ellipse cx={cx + 3.2} cy={cy + 2.5} rx="2.6" ry="5.8" transform={`rotate(28 ${cx + 3.2} ${cy + 2.5})`} opacity={1 - t * 0.2} />
      </g>
    );
  });
  return (
    <g>
      <path d={`M${x} ${H} Q${x + lean * 0.2} ${H - h * 0.5} ${top.x} ${top.y}`} stroke="#9b7130" strokeWidth="1.6" fill="none" />
      <ellipse cx={top.x} cy={top.y - 2} rx="2.2" ry="5" fill="#b5832a" />
      <g fill="#b5832a">{ears}</g>
      <path d={`M${x + lean * 0.1} ${H - h * 0.35} q 14 -6 22 -22`} stroke="#4f7a4f" strokeWidth="1.4" fill="none" />
    </g>
  );
}

export default function FieldPlate() {
  const rows = 18;
  const xs = Array.from({ length: rows + 1 }, (_, i) => -420 + i * (1320 / rows));

  return (
    <figure className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Illustration of a ploughed field at sunrise">
        <defs>
          <clipPath id="plate">
            <rect x="0" y="0" width={W} height={H} rx="18" />
          </clipPath>
        </defs>
        <g clipPath="url(#plate)">
          <rect width={W} height={H} fill="#f7f4ec" />
          {/* sun */}
          <circle cx="344" cy="122" r="74" fill="none" stroke="#b5832a" strokeOpacity="0.18" />
          <circle cx="344" cy="122" r="56" fill="none" stroke="#b5832a" strokeOpacity="0.28" />
          <circle cx="344" cy="122" r="40" fill="#d9a54a" />
          {/* hills */}
          <path d={`M0 ${HORIZON} C 70 140, 150 146, 220 ${HORIZON - 6} S 380 150, ${W} ${HORIZON - 4} V ${HORIZON + 4} H0Z`} fill="#dfe3cf" />
          <path d={`M0 ${HORIZON} C 60 158, 120 160, 180 ${HORIZON} Z`} fill="#c7d2b3" />
          {/* tree */}
          <line x1="96" y1={HORIZON - 2} x2="96" y2={HORIZON - 22} stroke="#2e5e3e" strokeWidth="1.6" />
          <circle cx="96" cy={HORIZON - 30} r="12" fill="#2e5e3e" />
          <circle cx="110" cy={HORIZON - 22} r="7" fill="#3f7a52" />
          {/* furrows */}
          {xs.slice(0, -1).map((x, i) => (
            <polygon
              key={i}
              points={`${VX},${HORIZON} ${x},${H} ${xs[i + 1]},${H}`}
              fill={i % 2 ? "#e6dcc4" : "#d3dbbd"}
            />
          ))}
          {xs.map((x, i) => (
            <line key={i} x1={VX} y1={HORIZON} x2={x} y2={H} stroke="#17241b" strokeOpacity="0.16" strokeWidth="0.8" />
          ))}
          <line x1="0" x2={W} y1={HORIZON} y2={HORIZON} stroke="#17241b" strokeOpacity="0.35" />
          {/* foreground wheat */}
          {stalk(392, 122, -10)}
          {stalk(414, 150, 6)}
          {stalk(438, 112, 16)}
          {/* seedling */}
          <path d="M70 360 V 326" stroke="#2e5e3e" strokeWidth="1.8" />
          <path d="M70 336 c 0 -14 10 -22 24 -22 c 0 14 -10 22 -24 22 Z" fill="#3f7a52" />
          <path d="M70 342 c 0 -11 -8 -18 -19 -18 c 0 11 8 18 19 18 Z" fill="#2e5e3e" />
        </g>
        <rect x="0.75" y="0.75" width={W - 1.5} height={H - 1.5} rx="17.5" fill="none" stroke="#d9d2c2" strokeWidth="1.5" />
      </svg>
      <figcaption className="num mt-2.5 flex justify-between text-[10.5px] uppercase tracking-[0.14em] text-faint">
        <span>Pl. 1 — furrow, horizon, harvest</span>
        <span>22 crops</span>
      </figcaption>
    </figure>
  );
}
