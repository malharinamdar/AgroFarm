"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Crops" },
  { href: "/fertilizer", label: "Fertilizer" },
  { href: "/method", label: "Method" },
];

export default function Header() {
  const path = usePathname();
  return (
    <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
      <Link href="/" className="flex items-center gap-2.5" aria-label="AgroFarm home">
        <Mark />
        <span className="font-display text-[22px] leading-none tracking-tight">AgroFarm</span>
      </Link>
      <nav className="flex items-center gap-1 text-sm">
        {LINKS.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`rounded-full px-3 py-1.5 transition-colors ${
                active ? "bg-ink text-paper" : "text-muted hover:text-ink"
              }`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}

function Mark() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden>
      <circle cx="13" cy="13" r="12.25" stroke="currentColor" strokeWidth="1.5" />
      <path d="M13 20V11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M13 13.5c0-3 2-5 5-5 0 3-2 5-5 5Z" fill="var(--color-leaf)" />
      <path d="M13 15.5c0-2.6-1.8-4.4-4.4-4.4 0 2.6 1.8 4.4 4.4 4.4Z" fill="var(--color-ochre)" />
    </svg>
  );
}
