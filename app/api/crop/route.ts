import { NextResponse } from "next/server";
import { parseCropInput, recommendCrops } from "@/lib/crop";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  const parsed = parseCropInput(body);
  if ("error" in parsed) return NextResponse.json(parsed, { status: 422 });
  return NextResponse.json(recommendCrops(parsed.value));
}
