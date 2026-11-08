import type { Experiment } from '../core/types';

/** Everything needed to reproduce what the viewer sees, encoded in the URL fragment (#s=…). */
export interface ShareState {
  v: 1;
  level: string;
  mode: 'solo' | 'versus';
  a: Experiment;
  b: Experiment | null;
}

const toB64Url = (bytes: Uint8Array) => {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const fromB64Url = (s: string) => {
  const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/'));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const res = new Response(new Blob([new Uint8Array(bytes)]).stream().pipeThrough(stream));
  return new Uint8Array(await res.arrayBuffer());
}

export async function encodeShare(state: ShareState): Promise<string> {
  const json = new TextEncoder().encode(JSON.stringify(state));
  if (typeof CompressionStream !== 'undefined') {
    try {
      return 'z' + toB64Url(await pipe(json, new CompressionStream('deflate-raw')));
    } catch {
      /* fall through to uncompressed */
    }
  }
  return 'j' + toB64Url(json);
}

export async function decodeShare(token: string): Promise<ShareState | null> {
  try {
    const kind = token[0];
    let bytes: Uint8Array = fromB64Url(token.slice(1));
    if (kind === 'z') bytes = await pipe(bytes, new DecompressionStream('deflate-raw'));
    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as ShareState;
    if (parsed.v !== 1 || !parsed.a) return null;
    return parsed;
  } catch {
    return null;
  }
}

export const readShareToken = () => {
  const m = /[#&]s=([A-Za-z0-9_-]+)/.exec(location.hash);
  return m ? m[1] : null;
};
