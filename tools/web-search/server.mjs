#!/usr/bin/env node
// MCP server with two tools: search (via the local SearXNG container) and read_page.
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { readPage } from "./page.mjs";
import { unansweredBy } from "./search.mjs";

const SEARXNG_URL = process.env.SEARXNG_URL ?? "http://127.0.0.1:8888";
const TIMEOUT_MS = 20_000;
const SNIPPET_CHARS = 300;

async function search(query, maxResults) {
  const url = new URL("/search", SEARXNG_URL);
  url.searchParams.set("q", query);
  url.searchParams.set("format", "json");

  const response = await fetch(url, { signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!response.ok) {
    throw new Error(`SearXNG returned HTTP ${response.status}`);
  }
  const body = await response.json();
  const blocked = unansweredBy(body);
  if (blocked) {
    throw Object.assign(new Error(blocked), { blocked: true });
  }
  return (body.results ?? []).slice(0, maxResults).map((result) => ({
    title: result.title,
    url: result.url,
    snippet: (result.content ?? "").slice(0, SNIPPET_CHARS),
    published: result.publishedDate ?? undefined,
  }));
}

function formatResults(results) {
  return results
    .map((r, i) => {
      const date = r.published ? `\nPublished: ${r.published}` : "";
      return `${i + 1}. ${r.title}\nURL: ${r.url}${date}\n${r.snippet}`;
    })
    .join("\n\n");
}

const server = new McpServer({ name: "web", version: "1.0.0" });

server.registerTool(
  "read_page",
  {
    description:
      "Read a public web page as clean text, one section at a time. If the result says " +
      "there is more, call again with the given offset only if you need it.",
    inputSchema: {
      url: z.string().url().describe("The http or https URL to read"),
      offset: z.number().int().min(0).default(0).describe("Character position to start from"),
    },
  },
  async ({ url, offset }) => {
    try {
      return { content: [{ type: "text", text: await readPage(url, offset) }] };
    } catch (error) {
      const text = `Could not read ${url}: ${error.message}. Try a different source.`;
      return { isError: true, content: [{ type: "text", text }] };
    }
  },
);

// OpenCode exposes tools as <server>_<tool>, so this one becomes web_search.
server.registerTool(
  "search",
  {
    description:
      "Search the web. Returns titles, URLs and short snippets. Snippets are not " +
      "enough to cite: read promising pages with web_read_page.",
    inputSchema: {
      query: z.string().min(1).describe("The search query"),
      max_results: z.number().int().min(1).max(10).default(5).describe("Results to return"),
    },
  },
  async ({ query, max_results }) => {
    try {
      const results = await search(query, max_results);
      const text = results.length
        ? formatResults(results)
        : "No results. Try a different query.";
      return { content: [{ type: "text", text }] };
    } catch (error) {
      const text = error.blocked
        ? error.message
        : `Search failed: ${error.message}. ` +
          `Is SearXNG running at ${SEARXNG_URL}? Start it with: docker compose up -d`;
      return { isError: true, content: [{ type: "text", text }] };
    }
  },
);

await server.connect(new StdioServerTransport());
