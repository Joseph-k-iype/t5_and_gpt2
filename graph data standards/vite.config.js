import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// SINGLE_FILE=1 inlines the whole app into one dist-single/index.html,
// which is what gets published as a shareable page. The default build
// is a normal multi-asset Vite build for hosting.
const single = process.env.SINGLE_FILE === "1";

export default defineConfig({
  plugins: [react(), ...(single ? [viteSingleFile()] : [])],
  base: "./",
  build: {
    target: "es2019",
    cssCodeSplit: !single,
    assetsInlineLimit: single ? 100000000 : 4096,
    chunkSizeWarningLimit: 1200,
  },
});
