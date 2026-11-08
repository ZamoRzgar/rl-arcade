import { useEffect } from 'react';
import { bootStore, useStore } from '../state/store';
import { decodeShare, readShareToken } from '../state/urlState';
import { TopBar } from './TopBar';
import { ControlDeck } from './ControlDeck';
import { Stage } from './Stage';
import { Insight } from './Insight';
import { HighScores } from './HighScores';
import { Help } from './Help';
import { usePlayback } from './playback';

export function App() {
  usePlayback();
  const toast = useStore((s) => s.toast);

  useEffect(() => {
    const token = readShareToken();
    if (token) {
      decodeShare(token).then((st) => {
        if (st) useStore.getState().loadShared(st.level, st.mode, st.a, st.b);
        else {
          bootStore();
          useStore.getState().flash('That share link could not be read, so the default level loaded instead.');
        }
      });
    } else bootStore();
  }, []);

  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => useStore.getState().set({ toast: null }), 3200);
    return () => clearTimeout(id);
  }, [toast]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      const s = useStore.getState();
      if (e.key === 'Escape' && s.help) s.set({ help: false });
      if (s.help) return;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          s.setPlaying(!s.playing);
          break;
        case 'ArrowRight':
          e.preventDefault();
          if (e.shiftKey) s.stepEpisode(1);
          else s.stepBy(1);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          if (e.shiftKey) s.stepEpisode(-1);
          else s.stepBy(-1);
          break;
        case 'Home':
          s.restart();
          break;
        case 'b':
        case 'B':
          s.togglePin();
          break;
        case 'd':
        case 'D':
          s.toggleOverlay('diff');
          break;
        case '1':
          s.setMode('solo');
          break;
        case '2':
          s.setMode('versus');
          break;
        case '+':
        case '=':
          s.setSpeed(s.speed + 1);
          break;
        case '-':
          s.setSpeed(s.speed - 1);
          break;
        case '?':
          s.set({ help: true });
          break;
        default:
          return;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="app">
      <TopBar />
      <div className="main">
        <ControlDeck />
        <Stage />
        <Insight />
      </div>
      <HighScores />
      <Help />
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}
