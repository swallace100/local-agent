// Fetches a web page and reduces it to readable text that fits a small context window.
import { BlockList, isIP } from "node:net";
import { lookup } from "node:dns/promises";
import { Readability } from "@mozilla/readability";
import { JSDOM, VirtualConsole } from "jsdom";
import TurndownService from "turndown";

export const PAGE_CHARS = 8_000;
const MAX_BYTES = 5_000_000;
const MAX_REDIRECTS = 5;
const TIMEOUT_MS = 20_000;
const USER_AGENT = "Mozilla/5.0 (compatible; local-research-agent/1.0)";

const privateRanges = new BlockList();
for (const [net, prefix] of [
  ["0.0.0.0", 8],
  ["10.0.0.0", 8],
  ["100.64.0.0", 10],
  ["127.0.0.0", 8],
  ["169.254.0.0", 16],
  ["172.16.0.0", 12],
  ["192.168.0.0", 16],
]) {
  privateRanges.addSubnet(net, prefix, "ipv4");
}
privateRanges.addAddress("::", "ipv6");
privateRanges.addAddress("::1", "ipv6");
privateRanges.addSubnet("fc00::", 7, "ipv6");
privateRanges.addSubnet("fe80::", 10, "ipv6");

export function isPrivateAddress(address) {
  const mapped = address.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/i);
  if (mapped) return isPrivateAddress(mapped[1]);
  return privateRanges.check(address, isIP(address) === 6 ? "ipv6" : "ipv4");
}

// A web page could otherwise point the agent at services on this machine or network.
export async function assertPublicUrl(url) {
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Only http and https URLs are allowed, got ${url.protocol}`);
  }
  const host = url.hostname.replace(/^\[|\]$/g, "");
  const addresses = isIP(host)
    ? [host]
    : (await lookup(host, { all: true })).map((entry) => entry.address);
  if (addresses.some(isPrivateAddress)) {
    throw new Error(`Refusing to read a private network address: ${url.hostname}`);
  }
}

async function fetchPublic(startUrl) {
  let url = new URL(startUrl);
  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    await assertPublicUrl(url);
    const response = await fetch(url, {
      redirect: "manual",
      signal: AbortSignal.timeout(TIMEOUT_MS),
      headers: { "user-agent": USER_AGENT, accept: "text/html,text/plain;q=0.9,*/*;q=0.5" },
    });
    const location = response.headers.get("location");
    if (response.status >= 300 && response.status < 400 && location) {
      url = new URL(location, url);
      continue;
    }
    return { response, url };
  }
  throw new Error(`Too many redirects from ${startUrl}`);
}

export function htmlToText(html, url) {
  const dom = new JSDOM(html, { url, virtualConsole: new VirtualConsole() });
  const article = new Readability(dom.window.document).parse();
  const title = article?.title || dom.window.document.title || "";
  const content = article?.content ?? dom.window.document.body?.innerHTML ?? "";
  const turndown = new TurndownService({ headingStyle: "atx", codeBlockStyle: "fenced" });
  turndown.remove(["script", "style", "noscript", "img", "svg"]);
  return { title, text: turndown.turndown(content).replace(/\n{3,}/g, "\n\n").trim() };
}

export function sliceText(text, offset) {
  const chunk = text.slice(offset, offset + PAGE_CHARS);
  const end = offset + chunk.length;
  const next =
    end < text.length
      ? `\n\n[Showing characters ${offset}-${end} of ${text.length}. To read more, call again with offset=${end}.]`
      : "";
  return chunk + next;
}

export async function readPage(startUrl, offset = 0) {
  const { response, url } = await fetchPublic(startUrl);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${url}`);
  }
  const type = response.headers.get("content-type") ?? "";
  if (!/text\/|html|json|xml/i.test(type)) {
    throw new Error(`Cannot read this content type: ${type || "unknown"}`);
  }
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > MAX_BYTES) {
    throw new Error(`Page is too large to read (${bytes.byteLength} bytes)`);
  }
  const raw = new TextDecoder().decode(bytes);
  const { title, text } = /html/i.test(type)
    ? htmlToText(raw, url.href)
    : { title: "", text: raw };
  if (!text) {
    throw new Error("The page has no readable text. It may require JavaScript.");
  }
  return `Title: ${title}\nURL: ${url.href}\n\n${sliceText(text, offset)}`;
}
