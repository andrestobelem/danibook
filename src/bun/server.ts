import { Database } from "bun:sqlite";
import { join, resolve, sep } from "path";
import { existsSync, mkdirSync } from "fs";

const PORT = 7861;
const PYTHON_BACKEND = "http://127.0.0.1:7860";
const DATA_DIR = process.env.DANIBOOK_DATA_DIR || join(process.cwd(), "data");
const CACHE_DIR = join(DATA_DIR, "cache");
mkdirSync(CACHE_DIR, { recursive: true });
const DB_PATH = join(CACHE_DIR, "archive_fts5.sqlite");

// Open the shared ArchiveMemory SQLite FTS5 index (same schema & tokenizer as Python)
const db = new Database(DB_PATH);
db.run("PRAGMA busy_timeout = 3000;");
db.run(`
  CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    source_file,
    content,
    page_number UNINDEXED,
    section_title UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
  );
`);

function buildFtsQuery(raw: string): string {
  const tokens = (raw.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || []).filter((t) => t.length >= 2);
  if (tokens.length === 0) return "";
  return tokens.slice(0, 12).map((t) => `"${t}"`).join(" OR ");
}

console.log(`⚡ Bun Core Server v${Bun.version} initializing on port ${PORT}...`);

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    // 1. Health & Identity
    if (url.pathname === "/api/health") {
      return Response.json({
        status: "ok",
        runtime: "bun",
        version: Bun.version,
        fts_active: true,
      });
    }

    // 2. Ultra-fast Native FTS5 Search Endpoint (shared ArchiveMemory index)
    if (url.pathname === "/api/fts/search") {
      const q = url.searchParams.get("q") || "";
      const ftsQuery = buildFtsQuery(q);
      if (!ftsQuery) return Response.json({ query: q, count: 0, results: [] });

      try {
        const startTime = performance.now();
        const stmt = db.query(`
          SELECT
            source_file,
            page_number,
            section_title,
            snippet(chunks_fts, 1, '<b>', '</b>', '...', 24) as snippet,
            -bm25(chunks_fts) as score
          FROM chunks_fts
          WHERE chunks_fts MATCH ?
          ORDER BY bm25(chunks_fts)
          LIMIT 10;
        `);
        const results = stmt.all(ftsQuery);
        const latencyMs = (performance.now() - startTime).toFixed(3);

        return Response.json({
          query: q,
          latency_ms: latencyMs,
          count: results.length,
          results,
        });
      } catch (err: any) {
        return Response.json({ error: "FTS5 search failed", details: err.message }, { status: 500 });
      }
    }

    // 3. Reverse-proxy SSE streams and API endpoints directly to the Metal-backed engine
    if (url.pathname.startsWith("/api/") && url.pathname !== "/api/health" && url.pathname !== "/api/fts/search") {
      const targetUrl = `${PYTHON_BACKEND}${url.pathname}${url.search}`;
      try {
        const proxyRes = await fetch(targetUrl, {
          method: req.method,
          headers: req.headers,
          body: req.method !== "GET" && req.method !== "HEAD" ? await req.blob() : undefined,
        });

        return new Response(proxyRes.body, {
          status: proxyRes.status,
          headers: proxyRes.headers,
        });
      } catch (err: any) {
        return Response.json({ error: "Inference engine offline", details: err.message }, { status: 502 });
      }
    }

    // 4. Serve vendor JS assets safely (prevent path traversal)
    if (url.pathname.startsWith("/vendor/")) {
      const vendorDir = resolve(process.cwd(), "src", "danibook", "ui", "vendor");
      const fileName = url.pathname.slice("/vendor/".length);
      const vendorPath = resolve(vendorDir, fileName);
      if (vendorPath.startsWith(vendorDir + sep) && existsSync(vendorPath)) {
        return new Response(Bun.file(vendorPath), {
          headers: { "Content-Type": "application/javascript; charset=utf-8" },
        });
      }
      return new Response("Not Found", { status: 404 });
    }

    // 5. Serve NotebookLM Material 3 static web application
    if (url.pathname === "/" || url.pathname === "/notebooklm.html") {
      const htmlPath = join(process.cwd(), "src", "danibook", "ui", "notebooklm.html");
      if (existsSync(htmlPath)) {
        return new Response(Bun.file(htmlPath), {
          headers: { "Content-Type": "text/html; charset=utf-8" },
        });
      }
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`🚀 Danibook Bun Server running at http://127.0.0.1:${server.port}`);
