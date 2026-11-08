import { LEVELS, getLevel } from '../core/envs/presets';
import { useStore } from '../state/store';
import { encodeShare } from '../state/urlState';
import { sfx } from '../design/sfx';
import { invalidateTheme } from '../viz/theme';

export function TopBar({ onHome }: { onHome: () => void }) {
  const levelId = useStore((s) => s.levelId);
  const mode = useStore((s) => s.mode);
  const sound = useStore((s) => s.sound);
  const crt = useStore((s) => s.crt);
  const palette = useStore((s) => s.palette);
  const st = useStore.getState();
  const level = getLevel(levelId);

  const share = async () => {
    const s = useStore.getState();
    const token = await encodeShare({ v: 1, level: s.levelId, mode: s.mode, a: s.slots[0].exp, b: s.mode === 'versus' ? s.slots[1].exp : null });
    const url = `${location.origin}${location.pathname}#s=${token}`;
    history.replaceState(null, '', `#s=${token}`);
    try {
      await navigator.clipboard.writeText(url);
      st.flash('Link copied. It reproduces exactly these runs.');
    } catch {
      st.flash('The link is in the address bar. Copy it from there.');
    }
  };

  return (
    <header className="topbar">
      <button type="button" className="logo" onClick={onHome} title="Back to the intro page">
        RL <span>ARCADE</span>
      </button>
      <div className="level-select">
        <label htmlFor="level" className="section-title" style={{ margin: 0 }}>
          Level
        </label>
        <select id="level" value={levelId} onChange={(e) => st.loadLevel(e.target.value)}>
          {LEVELS.map((l, i) => (
            <option key={l.id} value={l.id}>
              {i + 1}. {l.name}
            </option>
          ))}
        </select>
      </div>
      <div className="seg" role="group" aria-label="Players">
        <button type="button" className="btn" aria-pressed={mode === 'solo'} onClick={() => st.setMode('solo')} title="One algorithm, compared with your previous run (key 1)">
          1P Solo
        </button>
        <button type="button" className="btn" aria-pressed={mode === 'versus'} onClick={() => st.setMode('versus')} title="Two algorithms side by side, same level and seed (key 2)">
          2P Versus
        </button>
      </div>
      <span className="spacer" />
      <button type="button" className="btn" onClick={share} title="Copy a link that reproduces this exact setup">
        Share
      </button>
      <button
        type="button"
        className="btn"
        aria-pressed={sound}
        onClick={() => {
          if (!sound) {
            sfx.unlock();
            sfx.insert();
          }
          st.set({ sound: !sound });
        }}
        title="8-bit sound effects"
      >
        {sound ? 'Sound on' : 'Sound off'}
      </button>
      <button type="button" className="btn" aria-pressed={crt} onClick={() => st.set({ crt: !crt })} title="Scanline screen effect">
        CRT
      </button>
      <button
        type="button"
        className="btn"
        aria-pressed={palette === 'safe'}
        onClick={() => {
          const next = palette === 'safe' ? 'neon' : 'safe';
          document.documentElement.dataset.palette = next;
          invalidateTheme();
          st.set({ palette: next });
        }}
        title="Colorblind-safe palette (blue / orange)"
      >
        CB-safe
      </button>
      <button type="button" className="btn" onClick={onHome} title="What this site is and how to use it">
        Guide
      </button>
      <button type="button" className="btn icon" onClick={() => st.set({ help: true })} aria-label="Help and keyboard shortcuts" title="Help (?)">
        ?
      </button>
      <p className="challenge">
        <span style={{ color: 'var(--hi)' }}>CHALLENGE ›</span> {level.challenge}
      </p>
    </header>
  );
}
