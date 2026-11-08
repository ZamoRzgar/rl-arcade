import { useEffect, useRef, useState } from 'react';
import type { BackupNode, BackupSpec } from '../core/types';

interface DrawNode extends BackupNode {
  o: number;
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const ease = (t: number) => 1 - Math.pow(1 - t, 3);

/** Backup diagram that morphs between algorithms: shared nodes slide, others fade in or out. */
export function BackupDiagram({ spec }: { spec: BackupSpec }) {
  const [nodes, setNodes] = useState<DrawNode[]>(() => spec.nodes.map((n) => ({ ...n, o: 1 })));
  const current = useRef(nodes);
  current.current = nodes;
  const [shown, setShown] = useState(spec);

  useEffect(() => {
    const from = current.current;
    const to = spec.nodes;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const ids = new Set([...from.map((n) => n.id), ...to.map((n) => n.id)]);
    const start = performance.now();
    const dur = reduce ? 1 : 650;
    let raf = 0;
    setShown(spec);
    const tick = (now: number) => {
      const t = ease(Math.min(1, (now - start) / dur));
      const out: DrawNode[] = [];
      for (const id of ids) {
        const a = from.find((n) => n.id === id);
        const b = to.find((n) => n.id === id);
        if (a && b) out.push({ ...b, x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), o: lerp(a.o, 1, t) });
        else if (b) out.push({ ...b, o: t });
        else if (a && t < 1) out.push({ ...a, o: a.o * (1 - t) });
      }
      setNodes(out);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [spec]);

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edgeOpacity = (a?: DrawNode, b?: DrawNode) => (a && b ? Math.min(a.o, b.o) : 0);
  return (
    <figure className="backup" style={{ margin: 0 }}>
      <svg viewBox="0 0 100 100" role="img" aria-label={`Backup diagram: ${shown.caption}`}>
        {shown.edges.map(([u, v]) => {
          const a = byId.get(u);
          const b = byId.get(v);
          if (!a || !b) return null;
          return <line key={`${u}-${v}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="var(--text-dim)" strokeWidth={0.7} opacity={edgeOpacity(a, b)} />;
        })}
        {(shown.dashed ?? []).map(([u, v]) => {
          const a = byId.get(u);
          const b = byId.get(v);
          if (!a || !b) return null;
          return <line key={`d-${u}-${v}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="var(--neon-cyan)" strokeWidth={0.7} strokeDasharray="2 1.5" opacity={edgeOpacity(a, b)} />;
        })}
        {(shown.arcs ?? []).map((arc) => {
          const kids = arc.over.map((id) => byId.get(id)).filter((n): n is DrawNode => !!n);
          if (kids.length < 2) return null;
          const parentId = shown.edges.find(([, v]) => v === arc.over[0])?.[0];
          const p = parentId ? byId.get(parentId) : undefined;
          if (!p) return null;
          const pts = kids.map((k) => ({ x: lerp(p.x, k.x, 0.4), y: lerp(p.y, k.y, 0.4) }));
          const first = pts[0];
          const last = pts[pts.length - 1];
          const midY = Math.max(...pts.map((q) => q.y)) + 4;
          const o = Math.min(...kids.map((k) => k.o), p.o);
          return (
            <g key={arc.id} opacity={o}>
              <path d={`M ${first.x} ${first.y} Q ${(first.x + last.x) / 2} ${midY} ${last.x} ${last.y}`} fill="none" stroke="var(--accent)" strokeWidth={0.8} />
              <text x={(first.x + last.x) / 2} y={midY + 4} textAnchor="middle" fontSize={4.2} fill="var(--accent)" fontFamily="var(--font-body)">
                {arc.label}
              </text>
            </g>
          );
        })}
        {nodes.map((n) => {
          if (n.kind === 'state') return <circle key={n.id} cx={n.x} cy={n.y} r={4.2} fill="var(--screen)" stroke="var(--text)" strokeWidth={0.9} opacity={n.o} />;
          if (n.kind === 'action') return <circle key={n.id} cx={n.x} cy={n.y} r={2.4} fill="var(--accent)" opacity={n.o} />;
          if (n.kind === 'terminal') return <rect key={n.id} x={n.x - 3.5} y={n.y - 3.5} width={7} height={7} fill="var(--good)" opacity={n.o} />;
          if (n.kind === 'model')
            return (
              <g key={n.id} opacity={n.o}>
                <rect x={n.x - 7} y={n.y - 5} width={14} height={10} rx={1.5} fill="var(--panel-2)" stroke="var(--neon-cyan)" strokeWidth={0.7} />
                <text x={n.x} y={n.y + 1.6} textAnchor="middle" fontSize={5} fill="var(--neon-cyan)" fontFamily="var(--font-body)">
                  {n.label ?? 'M'}
                </text>
              </g>
            );
          return (
            <text key={n.id} x={n.x} y={n.y + 2} textAnchor="middle" fontSize={7} fill="var(--text-dim)" opacity={n.o}>
              ⋮
            </text>
          );
        })}
      </svg>
      <figcaption className="small muted">{shown.caption}</figcaption>
    </figure>
  );
}
