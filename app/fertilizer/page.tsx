import type { Metadata } from "next";
import FertilizerTool from "@/components/FertilizerTool";
import { FERT_CROPS, SOILS } from "@/lib/fertilizer";

export const metadata: Metadata = { title: "Fertilizer" };

export default function FertilizerPage() {
  return (
    <>
      <section className="pt-8 pb-12 sm:pt-14 sm:pb-16">
        <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">
          Fertilizer recommendation <span className="text-faint">·</span> 7 grades
        </p>
        <h1 className="mt-5 max-w-3xl font-display text-[44px] leading-[1.02] tracking-[-0.025em] sm:text-[68px]">
          Which bag goes on the <em className="text-leaf">field?</em>
        </h1>
        <p className="mt-6 max-w-xl text-[17px] leading-relaxed text-muted">
          Pick the soil and crop, then enter the soil test. A decision tree matches it to one of the seven common
          N–P–K grades.
        </p>
      </section>
      <FertilizerTool soils={SOILS} crops={FERT_CROPS} />
    </>
  );
}
