import { NextResponse } from "next/server";
import { parseFertInput, recommendFertilizer } from "@/lib/fertilizer";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  const parsed = parseFertInput(body);
  if ("error" in parsed) return NextResponse.json(parsed, { status: 422 });
  return NextResponse.json(recommendFertilizer(parsed.value));
}
