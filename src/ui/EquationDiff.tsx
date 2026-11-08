import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';
import type { AlgoMeta, EqToken } from '../core/types';

const html = (tex: string) => katex.renderToString(tex, { throwOnError: false, displayMode: false });
const memo = new Map<string, string>();
const render = (tex: string) => {
  let h = memo.get(tex);
  if (!h) {
    h = html(tex);
    memo.set(tex, h);
  }
  return h;
};

interface Gone {
  key: string;
  tex: string;
  x: number;
  y: number;
}

/**
 * The update rule, as tokens. Compared with `from` (the baseline / other player's algorithm):
 * tokens only in `to` glow as added, tokens whose text changed are yellow, removed tokens fade out
 * at their old position. Shared tokens slide to their new place (FLIP animation).
 */
export function EquationDiff({ from, to, hotParam }: { from: AlgoMeta | null; to: AlgoMeta; hotParam: string | null }) {
  const box = useRef<HTMLDivElement>(null);
  const prevRects = useRef(new Map<string, DOMRect>());
  const prevTokens = useRef<EqToken[]>(to.equation);
  const [gone, setGone] = useState<Gone[]>([]);
  const shared = from ? to.equation.filter((t) => from.equation.some((u) => u.id === t.id)).length : 0;
  // two rules with almost nothing in common (e.g. a bandit rule vs UCB's argmax) are shown side by side, not diffed
  const related = !!from && shared >= 4;
  const fromIds = useMemo(() => new Map((from?.equation ?? []).map((t) => [t.id, t.tex])), [from]);
  const removed = from && related ? from.equation.filter((t) => !to.equation.some((u) => u.id === t.id)) : [];

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const base = el.getBoundingClientRect();
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const nodes = el.querySelectorAll<HTMLElement>('[data-tok]');
    const nextRects = new Map<string, DOMRect>();
    nodes.forEach((n) => nextRects.set(n.dataset.tok!, n.getBoundingClientRect()));
    const changedAlgo = prevTokens.current !== to.equation;
    if (changedAlgo && !reduce) {
      nodes.forEach((n) => {
        const id = n.dataset.tok!;
        const old = prevRects.current.get(id);
        const now = nextRects.get(id)!;
        if (old) {
          n.animate([{ transform: `translate(${old.left - now.left}px, ${old.top - now.top}px)` }, { transform: 'none' }], { duration: 520, easing: 'cubic-bezier(.2,.8,.2,1)' });
        } else {
          n.animate([{ opacity: 0, transform: 'translateY(8px) scale(1.3)' }, { opacity: 1, transform: 'none' }], { duration: 520, easing: 'ease-out' });
        }
      });
      const lost: Gone[] = [];
      for (const t of prevTokens.current) {
        if (!to.equation.some((u) => u.id === t.id)) {
          const r = prevRects.current.get(t.id);
          if (r) lost.push({ key: `${t.id}-${Date.now()}`, tex: t.tex, x: r.left - base.left, y: r.top - base.top });
        }
      }
      if (lost.length) {
        setGone(lost);
        window.setTimeout(() => setGone([]), 1700);
      }
    }
    prevRects.current = nextRects;
    prevTokens.current = to.equation;
  }, [to.equation]);

  return (
    <div>
      <div className="eq" ref={box} aria-label={`Update rule of ${to.name}`}>
        {to.equation.map((t) => {
          const cls = related ? (!fromIds.has(t.id) ? 'added' : fromIds.get(t.id) !== t.tex ? 'changed' : '') : '';
          const hot = hotParam && t.param === hotParam ? ' hot' : '';
          return <span key={t.id} data-tok={t.id} className={`tok ${cls}${hot}`} dangerouslySetInnerHTML={{ __html: render(t.tex) }} />;
        })}
        {gone.map((g) => (
          <span key={g.key} className="tok gone" style={{ left: g.x, top: g.y }} dangerouslySetInnerHTML={{ __html: render(g.tex) }} />
        ))}
      </div>
      {from && !related && (
        <div className="eq-removed" style={{ color: 'var(--text-dim)' }}>
          A different kind of rule from {from.short}: <span dangerouslySetInnerHTML={{ __html: render(from.equation.map((t) => t.tex).join(' ')) }} />
        </div>
      )}
      {from && removed.length > 0 && (
        <div className="eq-removed">
          Removed from {from.short}:{' '}
          {removed.map((t) => (
            <span key={t.id} style={{ textDecoration: 'line-through', marginRight: 8 }} dangerouslySetInnerHTML={{ __html: render(t.tex) }} />
          ))}
        </div>
      )}
    </div>
  );
}
