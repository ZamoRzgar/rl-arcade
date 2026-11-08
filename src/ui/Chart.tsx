import { useEffect, useRef, useState } from 'react';

export interface Series {
  values: ArrayLike<number>;
  /** CSS color or var(--token) */
  color: string;
  label: string;
  dashed?: boolean;
  width?: number;
  alpha?: number;
}

interface Props {
  series: Series[];
  /** shade the area between series[i] and series[j] (green where i ≥ j) */
  fillBetween?: [number, number];
  higherIsBetter?: boolean;
  cursor?: number;
  markers?: number[];
  height?: number;
  logY?: boolean;
  xLabel: string;
  yLabel: string;
  onPick?: (x: number) => void;
  /** formatter for x tick labels */
  xFormat?: (x: number) => string;
  ariaLabel: string;
}

const resolve = (el: Element, c: string) => {
  const m = /var\((--[^)]+)\)/.exec(c);
  return m ? getComputedStyle(el).getPropertyValue(m[1]).trim() || '#fff' : c;
};

function niceTicks(lo: number, hi: number, n = 4): number[] {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => span / s <= n + 0.5) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toPrecision(10)));
  return out;
}

const fmtNum = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1000) return `${(v / 1000).toFixed(a >= 10000 ? 0 : 1)}k`;
  if (a >= 10 || v === 0) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toPrecision(2);
};

export function Chart({ series, fillBetween, higherIsBetter = true, cursor, markers = [], height = 170, logY = false, xLabel, yLabel, onPick, xFormat, ariaLabel }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(320);
  const [hover, setHover] = useState<{ x: number; px: number } | null>(null);
  const n = Math.max(0, ...series.map((s) => s.values.length));

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => setW(Math.floor(e[0].contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const padL = 44;
  const padR = 10;
  const padT = 10;
  const padB = 26;
  const tr = (v: number) => (logY ? Math.log10(Math.max(1e-12, v)) : v);

  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) for (let i = 0; i < s.values.length; i++) {
    const v = tr(s.values[i]);
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) {
    lo = 0;
    hi = 1;
  }
  if (hi - lo < 1e-9) {
    hi += 0.5;
    lo -= 0.5;
  }
  const pad = (hi - lo) * 0.06;
  lo -= pad;
  hi += pad;
  const X = (i: number) => padL + (n <= 1 ? 0 : (i / (n - 1)) * (w - padL - padR));
  const Y = (v: number) => padT + (1 - (tr(v) - lo) / (hi - lo)) * (height - padT - padB);

  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = w * dpr;
    c.height = height * dpr;
    c.style.height = `${height}px`;
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const col = (s: string) => resolve(c, s);
    ctx.clearRect(0, 0, w, height);
    ctx.fillStyle = col('var(--screen)');
    ctx.fillRect(padL, padT, w - padL - padR, height - padT - padB);
    // grid + y ticks
    ctx.font = `15px ${col('var(--font-data)') || 'monospace'}`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    const ticks = niceTicks(lo, hi);
    for (const tv of ticks) {
      const y = padT + (1 - (tv - lo) / (hi - lo)) * (height - padT - padB);
      ctx.strokeStyle = col('var(--line)');
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, Math.round(y) + 0.5);
      ctx.lineTo(w - padR, Math.round(y) + 0.5);
      ctx.stroke();
      ctx.fillStyle = col('var(--text-dim)');
      ctx.fillText(logY ? `1e${tv}` : fmtNum(tv), padL - 5, y);
    }
    // x ticks
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    if (n > 1) {
      for (const xv of niceTicks(0, n - 1, 5)) {
        if (xv < 0 || xv > n - 1) continue;
        ctx.fillStyle = col('var(--text-dim)');
        ctx.fillText(xFormat ? xFormat(xv) : fmtNum(xv + 1), X(xv), height - padB + 4);
      }
    }
    // fill between
    if (fillBetween) {
      const a = series[fillBetween[0]]?.values;
      const b = series[fillBetween[1]]?.values;
      if (a && b) {
        const m = Math.min(a.length, b.length);
        for (let i = 1; i < m; i++) {
          const better = higherIsBetter ? a[i] >= b[i] : a[i] <= b[i];
          ctx.fillStyle = col(better ? 'var(--good)' : 'var(--bad)');
          ctx.globalAlpha = 0.18;
          ctx.beginPath();
          ctx.moveTo(X(i - 1), Y(a[i - 1]));
          ctx.lineTo(X(i), Y(a[i]));
          ctx.lineTo(X(i), Y(b[i]));
          ctx.lineTo(X(i - 1), Y(b[i - 1]));
          ctx.closePath();
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
    }
    // markers (live tweaks)
    for (const m of markers) {
      ctx.strokeStyle = col('var(--hi)');
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(X(m), padT);
      ctx.lineTo(X(m), height - padB);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // series
    for (const s of series) {
      if (s.values.length === 0) continue;
      ctx.strokeStyle = col(s.color);
      ctx.globalAlpha = s.alpha ?? 1;
      ctx.lineWidth = s.width ?? 2;
      ctx.setLineDash(s.dashed ? [5, 4] : []);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.values.length; i++) {
        const v = s.values[i];
        if (!Number.isFinite(tr(v))) continue;
        if (!started) {
          ctx.moveTo(X(i), Y(v));
          started = true;
        } else ctx.lineTo(X(i), Y(v));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      // emphasized endpoint
      const last = s.values.length - 1;
      if (!s.dashed && last >= 0 && Number.isFinite(tr(s.values[last]))) {
        ctx.fillStyle = col(s.color);
        ctx.fillRect(X(last) - 3, Y(s.values[last]) - 3, 6, 6);
      }
    }
    // playhead
    if (cursor !== undefined && n > 1) {
      ctx.strokeStyle = col('var(--text)');
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.moveTo(X(cursor) + 0.5, padT);
      ctx.lineTo(X(cursor) + 0.5, height - padB);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // axis labels
    ctx.fillStyle = col('var(--text-dim)');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(xLabel, w - padR, height - 1);
    ctx.save();
    ctx.translate(11, padT + (height - padT - padB) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  });

  const pick = (clientX: number) => {
    const c = cv.current;
    if (!c || n < 1) return null;
    const rect = c.getBoundingClientRect();
    const px = clientX - rect.left;
    const x = Math.round(((px - padL) / Math.max(1, w - padL - padR)) * (n - 1));
    return { x: Math.max(0, Math.min(n - 1, x)), px };
  };

  return (
    <div className="chart-wrap" ref={wrap}>
      <canvas
        ref={cv}
        role="img"
        aria-label={ariaLabel}
        style={{ cursor: onPick ? 'pointer' : 'default' }}
        onPointerMove={(e) => setHover(pick(e.clientX))}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => {
          const p = pick(e.clientX);
          if (p && onPick) onPick(p.x);
        }}
      />
      {hover && (
        <div className="chart-tip" style={{ left: Math.min(Math.max(0, hover.px + 10), Math.max(0, w - 170)), top: 6 }}>
          <div className="muted">
            {xLabel} {xFormat ? xFormat(hover.x) : hover.x + 1}
          </div>
          {series.map((s) =>
            hover.x < s.values.length ? (
              <div key={s.label} style={{ color: resolveCss(s.color) }}>
                {s.label}: {fmtNum(s.values[hover.x])}
              </div>
            ) : null,
          )}
        </div>
      )}
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i style={{ background: s.dashed ? `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)` : s.color, opacity: s.alpha ?? 1 }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

const resolveCss = (c: string) => c;
