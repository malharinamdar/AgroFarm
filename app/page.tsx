import CropTool from "@/components/CropTool";
import FieldPlate from "@/components/FieldPlate";
import { CROPS } from "@/lib/fields";
import { medianConditions } from "@/lib/crop";

const PRESETS = [
  { label: "Paddy field", crop: "rice" },
  { label: "Cotton plot", crop: "cotton" },
  { label: "Chickpea field", crop: "chickpea" },
  { label: "Apple orchard", crop: "apple" },
];

export default function Home() {
  const presets = PRESETS.map((p) => ({ label: p.label, values: medianConditions(p.crop) }));
  const allCrops = Object.values(CROPS)
    .map((c) => c.name)
    .sort();

  return (
    <>
      <section className="grid items-center gap-10 pt-8 pb-12 sm:pt-14 sm:pb-16 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] lg:gap-14">
        <div>
        <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">
          Crop recommendation <span className="text-faint">·</span> 22 crops{" "}
          <span className="text-faint">·</span> 2,200 field records
        </p>
        <h1 className="mt-5 max-w-3xl font-display text-[44px] leading-[1.02] tracking-[-0.025em] sm:text-[64px]">
          What should this field grow <em className="text-leaf">next?</em>
        </h1>
        <p className="mt-6 max-w-xl text-[17px] leading-relaxed text-muted">
          Enter a soil test and the local climate. A random forest ranks every crop it knows, then shows how your
          field compares with the conditions each one is actually grown in.
        </p>
        </div>
        <div className="mx-auto w-full max-w-md lg:max-w-none">
          <FieldPlate />
        </div>
      </section>
      <CropTool presets={presets} allCrops={allCrops} />
    </>
  );
}
