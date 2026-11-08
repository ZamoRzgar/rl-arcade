import { useMemo } from 'react';
import type { AlgoMeta, PseudoLine } from '../core/types';

type Row = { kind: 'same' | 'add' | 'del'; line: PseudoLine; toIndex: number };

/** Line diff (longest common subsequence on the text) from `a` to `b`. */
export function diffLines(a: PseudoLine[], b: PseudoLine[]): Row[] {
  const n = a.length;
  const m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = a[i].text === b[j].text ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out: Row[] = [];
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i].text === b[j].text) {
      out.push({ kind: 'same', line: b[j], toIndex: j });
      i++;
      j++;
    } else if (j < m && (i >= n || dp[i][j + 1] >= dp[i + 1][j])) {
      out.push({ kind: 'add', line: b[j], toIndex: j });
      j++;
    } else {
      out.push({ kind: 'del', line: a[i], toIndex: -1 });
      i++;
    }
  }
  return out;
}

export function Pseudocode({ from, to, current, hotParam, showDiff }: { from: AlgoMeta | null; to: AlgoMeta; current: number; hotParam: string | null; showDiff: boolean }) {
  const rows = useMemo<Row[]>(
    () => (from && showDiff && from.id !== to.id ? diffLines(from.pseudocode, to.pseudocode) : to.pseudocode.map((line, i) => ({ kind: 'same' as const, line, toIndex: i }))),
    [from, to, showDiff],
  );
  return (
    <div className="code" role="list" aria-label={`Pseudocode of ${to.name}`}>
      {rows.map((r, k) => {
        const cur = r.toIndex >= 0 && r.toIndex === current;
        const hot = hotParam && r.line.params?.includes(hotParam);
        return (
          <div key={k} role="listitem" className={`ln ${r.kind}${cur ? ' cur' : ''}${hot ? ' hot' : ''}`} aria-current={cur ? 'step' : undefined}>
            <span className="g">{r.kind === 'add' ? '+' : r.kind === 'del' ? '−' : cur ? '▸' : ''}</span>
            <span>
              {'  '.repeat(r.line.indent)}
              {r.line.text}
            </span>
          </div>
        );
      })}
    </div>
  );
}
