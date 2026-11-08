import { useEffect, useRef } from 'react';
import { useStore } from '../state/store';

export function Help() {
  const open = useStore((s) => s.help);
  const ref = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (open) ref.current?.focus();
  }, [open]);
  if (!open) return null;
  const close = () => useStore.getState().set({ help: false });
  return (
    <div className="modal-back" role="presentation" onClick={close}>
      <div className="modal panel" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={(e) => e.stopPropagation()}>
        <h2 id="help-title">HOW TO PLAY</h2>
        <p>
          Every run is computed in full the moment you change something, using the same random seed. Turn any knob or swap the cartridge, and the
          previous run becomes the <b>baseline</b>: it shows as a faded ghost on the board, as dashed curves on the charts, and in the "What changed"
          chips. In <b>2P Versus</b>, two algorithms play the same level with the same luck.
        </p>
        <div className="keys">
          <kbd>Space</kbd>
          <span>Start / pause</span>
          <kbd>→</kbd>
          <span>Step forward (Shift: next episode)</span>
          <kbd>←</kbd>
          <span>Step back (Shift: previous episode)</span>
          <kbd>Home</kbd>
          <span>Back to episode 1</span>
          <kbd>B</kbd>
          <span>Pin / unpin the baseline</span>
          <kbd>D</kbd>
          <span>Toggle the DIFF layer</span>
          <kbd>1</kbd>
          <span>1P Solo</span>
          <kbd>2</kbd>
          <span>2P Versus</span>
          <kbd>+ / −</kbd>
          <span>Faster / slower</span>
          <kbd>?</kbd>
          <span>This help</span>
        </div>
        <p className="small muted">
          Grid cells: amber arrows show the greedy policy, colors show value (blue high, pink low), pink sparks are exploratory moves, and outlined
          cells are where the policy differs from the baseline. Values come from the real algorithms running in your browser. Nothing is pre-recorded.
        </p>
        <button ref={ref} type="button" className="btn primary" onClick={close}>
          Got it
        </button>
      </div>
    </div>
  );
}
