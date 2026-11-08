import { useId } from 'react';
import type { ParamDef, ParamValue } from '../core/types';

interface Props {
  def: ParamDef;
  value: ParamValue;
  changed: boolean;
  hot: boolean;
  onChange(v: ParamValue): void;
  onReset(): void;
  onHover(on: boolean): void;
  onSweep(): void;
}

const toSlider = (d: ParamDef, v: number) => {
  const lo = d.min ?? 0;
  const hi = d.max ?? 1;
  if (d.scale === 'log') return (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
  return (v - lo) / (hi - lo);
};
const fromSlider = (d: ParamDef, f: number) => {
  const lo = d.min ?? 0;
  const hi = d.max ?? 1;
  let v = d.scale === 'log' ? Math.exp(Math.log(lo) + f * (Math.log(hi) - Math.log(lo))) : lo + f * (hi - lo);
  if (d.scale === 'int') v = Math.round(v);
  else if (d.step) v = Math.round(v / d.step) * d.step;
  else v = Number(v.toPrecision(3));
  return Number(v.toFixed(6));
};

export function ParamControl({ def: d, value, changed, hot, onChange, onReset, onHover, onSweep }: Props) {
  const id = useId();
  const numeric = d.scale === 'linear' || d.scale === 'log' || d.scale === 'int';
  const isDefault = value === d.default;
  return (
    <div
      className={`param${changed ? ' changed' : ''}`}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      onFocus={() => onHover(true)}
      onBlur={() => onHover(false)}
    >
      <div className="param-row">
        <label className="param-label" htmlFor={id}>
          {d.symbol && <b>{d.symbol}</b>}
          {d.label}
        </label>
        {numeric && (
          <input
            id={id}
            type="number"
            value={Number(value)}
            min={d.min}
            max={d.max}
            step={d.scale === 'int' ? 1 : d.step ?? 'any'}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (Number.isFinite(v)) onChange(Math.max(d.min ?? -Infinity, Math.min(d.max ?? Infinity, d.scale === 'int' ? Math.round(v) : v)));
            }}
          />
        )}
        {d.scale === 'bool' && <input id={id} type="checkbox" checked={value === true} onChange={(e) => onChange(e.target.checked)} />}
        {d.scale === 'enum' && (
          <select id={id} value={String(value)} onChange={(e) => onChange(e.target.value)}>
            {d.options!.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        )}
      </div>
      {numeric && (
        <input
          type="range"
          aria-label={`${d.label} slider`}
          min={0}
          max={1000}
          value={Math.round(toSlider(d, Number(value)) * 1000)}
          onChange={(e) => onChange(fromSlider(d, Number(e.target.value) / 1000))}
        />
      )}
      <div className="param-row">
        {hot ? (
          <div className="explain">
            <div>▲ {d.explainUp}</div>
            <div>▼ {d.explainDown}</div>
          </div>
        ) : (
          <div className="explain" style={{ flex: 1 }} />
        )}
      </div>
      <div className="param-actions" style={{ marginTop: 4 }}>
        <button className="mini" type="button" onClick={onSweep} title={`Run the experiment across the whole range of ${d.label} and compare`}>
          Sweep
        </button>
        {!isDefault && (
          <button className="mini" type="button" onClick={onReset}>
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
