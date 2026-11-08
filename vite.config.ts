/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Relative base so the static build works on GitHub Pages sub-paths as well as Vercel.
  base: './',
  worker: { format: 'es' },
  build: {
    rollupOptions: {
      output: {
        // separate long-lived vendor chunks for better caching
        manualChunks: { react: ['react', 'react-dom', 'zustand'], katex: ['katex'] },
      },
    },
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    testTimeout: 60_000,
  },
});
