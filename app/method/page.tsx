import type { Metadata } from "next";
import metrics from "@/lib/model/metrics.json";

export const metadata: Metadata = { title: "Method" };

const LABELS: Record<string, string> = {
  N: "Nitrogen",
  P: "Phosphorus",
  K: "Potassium",
  temperature: "Temperature",
  humidity: "Humidity",
  ph: "Soil pH",
  rainfall: "Rainfall",
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function Method() {
  const { crop, fertilizer } = metrics;
  const importance = Object.entries(crop.importance).sort((a, b) => b[1] - a[1]);
  const maxImp = importance[0][1];

  return (
    <article className="pt-8 sm:pt-14">
      <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Method</p>
      <h1 className="mt-5 max-w-3xl font-display text-[40px] leading-[1.05] tracking-[-0.02em] sm:text-[56px]">
        How the recommendations are made
      </h1>

      <div className="mt-14 grid gap-14 lg:grid-cols-2">
        <section>
          <h2 className="font-display text-2xl">Crops</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            {crop.rows.toLocaleString()} field records, {crop.classes} crops, seven inputs: N, P, K, pH, temperature,
            humidity and rainfall. The first version of AgroFarm used logistic regression; it now uses a 100-tree
            random forest. Both are scored on the same held-out split ({crop.test_size} rows) the original notebook
            used.
          </p>
          <table className="mt-6 w-full text-[15px]">
            <tbody className="divide-y divide-line">
              <Row label="v1 · logistic regression" value={pct(crop.v1_logreg_accuracy)} />
              <Row label="v2 · random forest" value={pct(crop.rf_accuracy)} strong />
              <Row label="v2 · macro F1" value={pct(crop.rf_macro_f1)} />
              <Row
                label="v2 · 5-fold cross-validation"
                value={`${pct(crop.rf_cv5_mean)} ± ${(crop.rf_cv5_std * 100).toFixed(1)}`}
              />
            </tbody>
          </table>
          <p className="mt-3 text-[13px] text-faint">
            The served model is refit on all rows after scoring. Each record in this dataset is a single crop&apos;s
            conditions, so the forest learns which crop a field most resembles. It does not model yield or price.
          </p>
        </section>

        <section>
          <h2 className="font-display text-2xl">What the forest relies on</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            Mean decrease in impurity across the forest. Water (rainfall, humidity) separates crops more than
            nutrients do.
          </p>
          <ul className="mt-6 space-y-3">
            {importance.map(([k, v]) => (
              <li key={k} className="grid grid-cols-[110px_1fr_48px] items-center gap-4 text-[14px]">
                <span>{LABELS[k]}</span>
                <span className="h-2 rounded-full bg-paper-2">
                  <span className="block h-full rounded-full bg-leaf" style={{ width: `${(v / maxImp) * 100}%` }} />
                </span>
                <span className="num text-right text-xs text-muted">{v.toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </section>

        <section>
          <h2 className="font-display text-2xl">Fertilizer</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            A decision tree over soil type, crop, temperature, humidity, moisture and the N–P–K soil test.
            {" "}
            {fertilizer.rows} labelled rows across {fertilizer.classes} grades, so it is scored with 5-fold
            cross-validation rather than a single split.
          </p>
          <table className="mt-6 w-full text-[15px]">
            <tbody className="divide-y divide-line">
              <Row
                label="5-fold cross-validated accuracy"
                value={`${pct(fertilizer.dt_cv5_mean)} ± ${(fertilizer.dt_cv5_std * 100).toFixed(1)}`}
                strong
              />
            </tbody>
          </table>
        </section>

        <section>
          <h2 className="font-display text-2xl">Serving</h2>
          <p className="mt-3 max-w-prose leading-relaxed text-muted">
            Models are trained in scikit-learn (<code className="num text-[13px]">ml/train.py</code>) and exported
            as flat tree arrays. The Next.js API walks the trees directly, so there is no Python server; a parity
            check at export time asserts the exported forest matches scikit-learn&apos;s probabilities.
          </p>
          <p className="mt-4 text-[14px]">
            <a
              className="underline decoration-line underline-offset-4 hover:decoration-ink"
              href="https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"
            >
              Crop recommendation dataset
            </a>
            <span className="mx-2 text-faint">/</span>
            <a
              className="underline decoration-line underline-offset-4 hover:decoration-ink"
              href="https://github.com/malharinamdar/AgroFarm/blob/main/ml/train.py"
            >
              training script
            </a>
          </p>
        </section>
      </div>
    </article>
  );
}

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <tr>
      <td className={`py-2.5 ${strong ? "" : "text-muted"}`}>{label}</td>
      <td className={`num py-2.5 text-right ${strong ? "text-leaf" : ""}`}>{value}</td>
    </tr>
  );
}
