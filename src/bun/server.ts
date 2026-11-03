import { Database } from "bun:sqlite";
import { join } from "path";
import { existsSync, readFileSync } from "fs";

const PORT = 7861;
const PYTHON_BACKEND = "http://127.0.0.1:7860";
const DB_PATH = join(process.cwd(), "data", "danibook_fts.db");

// Initialize native SQLite with FTS5 for ultra-fast document search
const db = new Database(DB_PATH);
db.run(`
  CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks USING fts5(
    source_file,
    chunk_index UNINDEXED,
    content,
    tokenize = 'porter unicode61'
  );
`);

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

    // 2. Ultra-fast Native FTS5 Search Benchmark Endpoint
    if (url.pathname === "/api/fts/search") {
      const q = url.searchParams.get("q") || "";
      if (!q.trim()) return Response.json({ results: [] });

      const startTime = performance.now();
      const stmt = db.query(`
        SELECT source_file, chunk_index, snippet(document_chunks, 2, '<b>', '</b>', '...', 24) as snippet, rank
        FROM document_chunks
        WHERE document_chunks MATCH ?
        ORDER BY rank
        LIMIT 10;
      `);
      const results = stmt.all(q);
      const latencyMs = (performance.now() - startTime).toFixed(3);

      return Response.json({
        query: q,
        latency_ms: latencyMs,
        count: results.length,
        results,
      });
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

    // 4. Serve vendor JS assets
    if (url.pathname.startsWith("/vendor/")) {
      const fileName = url.pathname.replace("/vendor/", "");
      const vendorPath = join(process.cwd(), "src", "danibook", "ui", "vendor", fileName);
      if (existsSync(vendorPath)) {
        return new Response(readFileSync(vendorPath), {
          headers: { "Content-Type": "application/javascript; charset=utf-8" },
        });
      }
    }

    // 5. Serve NotebookLM Material 3 static web application
    if (url.pathname === "/" || url.pathname === "/notebooklm.html") {
      const htmlPath = join(process.cwd(), "src", "danibook", "ui", "notebooklm.html");
      if (existsSync(htmlPath)) {
        return new Response(readFileSync(htmlPath), {
          headers: { "Content-Type": "text/html; charset=utf-8" },
        });
      }
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`🚀 Danibook Bun Server running at http://127.0.0.1:${server.port}`);
