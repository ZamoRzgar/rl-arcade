import { algoMeta } from '../core/registry';
import { getLevel } from '../core/envs/presets';
import { useStore } from '../state/store';
import { expHash } from '../runner/client';

export function HighScores() {
  const history = useStore((s) => s.history);
  const levelId = useStore((s) => s.levelId);
  const current = useStore((s) => s.slots[0].exp);
  const st = useStore.getState();
  const rows = history.filter((h) => h.levelId === levelId);
  const curHash = expHash(current);
  const best = rows.length ? Math.max(...rows.map((r) => r.score)) : null;
  return (
    <section className="scores">
      <div className="panel">
        <h2 className="section-title">
          High scores · {getLevel(levelId).name}
          <span className="row">
            <span className="small muted" style={{ fontFamily: 'var(--font-body)', letterSpacing: 0, textTransform: 'none' }}>
              Click a run to restore it. Shift-click to load it as P2.
            </span>
            {history.length > 0 && (
              <button type="button" className="mini" onClick={() => {
                st.set({ history: [] });
                try {
                  localStorage.removeItem('rl-arcade-history-v1');
                } catch {
                  /* ignore */
                }
              }}>
                Clear
              </button>
            )}
          </span>
        </h2>
        {rows.length === 0 ? (
          <p className="small muted" style={{ margin: 0 }}>
            No runs on this level yet. Every run you make appears here with what changed and its score.
          </p>
        ) : (
          <div className="table-wrap">
            <table className="hs">
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Cartridge</th>
                  <th scope="col">Changed</th>
                  <th scope="col">Seed</th>
                  <th scope="col" style={{ textAlign: 'right' }}>
                    Score
                  </th>
                  <th scope="col" style={{ textAlign: 'right' }}>
                    Settled
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((h, i) => (
                  <tr
                    key={h.id}
                    className={h.hash === curHash ? 'current' : ''}
                    tabIndex={0}
                    onClick={(e) => st.restoreHistory(h.id, e.shiftKey)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') st.restoreHistory(h.id, e.shiftKey);
                    }}
                  >
                    <td>{rows.length - i}</td>
                    <td>
                      {algoMeta(h.algoId).short} {best !== null && h.score === best && rows.length > 1 && <span className="crown">★ BEST</span>}
                    </td>
                    <td>
                      {h.changes.map((c) => (
                        <span key={c} className="chg">
                          {c}
                        </span>
                      ))}
                    </td>
                    <td className="num">{h.exp.run.seed}</td>
                    <td className="num">{h.score.toFixed(Math.abs(h.score) >= 100 ? 0 : 2)}</td>
                    <td className="num">{h.isBandit || h.converged < 0 ? '—' : algoMeta(h.algoId).family === 'dp' ? `${h.converged} fr` : `ep ${h.converged + 1}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
