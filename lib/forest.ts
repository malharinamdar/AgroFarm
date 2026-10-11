// Inference for the forests exported by ml/train.py.
// sklearn casts inputs to float32 before walking a tree, so we do the same.

export type Tree = {
  f: number[];
  t: number[];
  l: number[];
  r: number[];
  v: ([number, number][] | null)[];
};

export function predictProba(trees: Tree[], x: number[], nClasses: number): number[] {
  const xf = x.map(Math.fround);
  const out = new Array<number>(nClasses).fill(0);
  for (const tree of trees) {
    let i = 0;
    while (tree.f[i] !== -1) i = xf[tree.f[i]] <= tree.t[i] ? tree.l[i] : tree.r[i];
    for (const [c, p] of tree.v[i] ?? []) out[c] += p;
  }
  return out.map((v) => v / trees.length);
}
