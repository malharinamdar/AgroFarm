import type { Metadata } from "next";
import { Fraunces, Geist, Geist_Mono } from "next/font/google";
import Header from "@/components/Header";
import SoilStrata from "@/components/SoilStrata";
import "./globals.css";

const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-fraunces", axes: ["opsz", "SOFT"] });
const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: { default: "AgroFarm — crop & fertilizer recommendations", template: "%s · AgroFarm" },
  description:
    "Rank 22 crops for a field from its soil nutrients and climate, and pick a fertilizer grade. Random forest trained on 2,200 field records.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${fraunces.variable} ${geist.variable} ${geistMono.variable}`}>
      <body className="min-h-dvh">
        <Header />
        <main className="mx-auto w-full max-w-6xl px-5 sm:px-8">{children}</main>
        <SoilStrata />
        <footer className="mx-auto w-full max-w-6xl px-5 pb-10 pt-2 sm:px-8">
          <div className="flex flex-col gap-2 pt-6 text-sm text-muted sm:flex-row sm:items-center sm:justify-between">
            <p>
              AgroFarm by{" "}
              <a className="text-ink underline decoration-line underline-offset-4 hover:decoration-ink" href="https://github.com/malharinamdar">
                Malhar Inamdar
              </a>
            </p>
            <p className="num text-xs">
              <a className="hover:text-ink" href="https://github.com/malharinamdar/AgroFarm">source</a>
              <span className="mx-2 text-faint">/</span>
              <a className="hover:text-ink" href="https://github.com/malharinamdar/AgroFarm/tree/legacy-streamlit">
                v1 (streamlit)
              </a>
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
