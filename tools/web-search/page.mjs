// Fetches a web page, PDF or Word document and reduces it to readable text that fits a
// small context window.
import { BlockList, isIP } from "node:net";
import { lookup as dnsLookup } from "node:dns";
import { lookup } from "node:dns/promises";
import { Readability } from "@mozilla/readability";
import { JSDOM, VirtualConsole } from "jsdom";
import mammoth from "mammoth";
import { extractText, getDocumentProxy, getMeta } from "unpdf";
import TurndownService from "turndown";
// undici's own fetch, so that it and the Agent below are the same version.
import { Agent, fetch } from "undici";

export const PAGE_CHARS = 8_000;
const MAX_BYTES = 5_000_000;
const MAX_DOCUMENT_BYTES = 25_000_000;
const RECENT_PAGES = 8;
const MAX_REDIRECTS = 5;
const TIMEOUT_MS = 20_000;
// Measured on 13 ordinary pages, Readability kept at least 57% of the text. On python.org's
// downloads page it kept 15% and lost the current version.
const MIN_ARTICLE_SHARE = 0.4;
const PAGE_FURNITURE = "script, style, noscript, template, svg, nav, header, footer, aside, form, iframe";
const USER_AGENT ="Mozilla/5.0 (compatible; local-research-agent/1.0)";

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

// Checks the addresses a connection is about to use. assertPublicUrl alone isn't enough,
// because a hostile DNS server can answer with a public address for that check and a
// private one for the connection that follows.
export function publicLookup(hostname, options, callback) {
  dnsLookup(hostname, { ...options, all: true }, (error, addresses) => {
    if (error) return callback(error);
    if (addresses.some((entry) => isPrivateAddress(entry.address))) {
      return callback(new Error(`Refusing to read a private network address: ${hostname}`));
    }
    if (options.all) return callback(null, addresses);
    callback(null, addresses[0].address, addresses[0].family);
  });
}

const publicOnly = new Agent({ connect: { lookup: publicLookup } });

async function fetchPublic(startUrl) {
  let url = new URL(startUrl);
  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    await assertPublicUrl(url);
    const response = await fetch(url, {
      dispatcher: publicOnly,
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

const squash = (text) => (text ?? "").replace(/\s+/g, " ").trim();

// The page's main content with the navigation and other furniture taken out.
function wholePage(document) {
  const root = (document.querySelector("main, [role=main]") ?? document.body)?.cloneNode(true);
  root?.querySelectorAll(PAGE_FURNITURE).forEach((node) => node.remove());
  return root;
}

export function htmlToText(html, url) {
  const dom = new JSDOM(html, { url, virtualConsole: new VirtualConsole() });
  // Copied first, because Readability changes the document it reads.
  const whole = wholePage(dom.window.document);
  const article = new Readability(dom.window.document).parse();
  const title = article?.title || dom.window.document.title || "";
  // Readability looks for one article. On a page that is mostly lists and tables, such as
  // a downloads page, it can keep a side note and drop the rest, so use the whole page then.
  const keptEnough =
    article && squash(article.textContent).length >= MIN_ARTICLE_SHARE * squash(whole?.textContent).length;
  const content = keptEnough ? article.content : (whole?.innerHTML ?? "");
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

// Servers often send documents as a generic download, so the file name is checked too.
export function documentKind(type, pathname) {
  const generic = /octet-stream|binary/i.test(type) || !type;
  if (/application\/pdf/i.test(type) || (generic && /\.pdf$/i.test(pathname))) return "pdf";
  if (/wordprocessingml\.document/i.test(type) || (generic && /\.docx$/i.test(pathname))) return "docx";
  return null;
}

export async function pdfToText(bytes) {
  const pdf = await getDocumentProxy(new Uint8Array(bytes));
  const { text: pages } = await extractText(pdf, { mergePages: false });
  const meta = await getMeta(pdf).catch(() => null);
  const text = pages
    .map((page, index) => `[Page ${index + 1}]\n${page.trim()}`)
    .filter((page) => /\n./.test(page))
    .join("\n\n");
  return { title: meta?.info?.Title?.trim() || "", text };
}

export async function docxToText(bytes) {
  const { value } = await mammoth.extractRawText({ buffer: Buffer.from(bytes) });
  return { title: "", text: value.replace(/\n{3,}/g, "\n\n").trim() };
}

async function fetchText(startUrl) {
  const { response, url } = await fetchPublic(startUrl);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${url}`);
  }
  const type = response.headers.get("content-type") ?? "";
  const kind = documentKind(type, url.pathname);
  if (!kind && !/text\/|html|json|xml/i.test(type)) {
    throw new Error(`Cannot read this content type: ${type || "unknown"}`);
  }
  const limit = kind ? MAX_DOCUMENT_BYTES : MAX_BYTES;
  // Checked before downloading when the server says, and again after in case it didn't.
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (declared > limit) {
    throw new Error(`Too large to read (${declared} bytes)`);
  }
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > limit) {
    throw new Error(`Too large to read (${bytes.byteLength} bytes)`);
  }
  let result;
  if (kind === "pdf") result = await pdfToText(bytes);
  else if (kind === "docx") result = await docxToText(bytes);
  else {
    const raw = new TextDecoder().decode(bytes);
    result = /html/i.test(type) ? htmlToText(raw, url.href) : { title: "", text: raw };
  }
  if (!result.text) {
    throw new Error(
      kind === "pdf"
        ? "This PDF has no text in it. It is probably a scan."
        : "The page has no readable text. It may require JavaScript.",
    );
  }
  return { ...result, url: url.href };
}

// Reading the next section of a page shouldn't download and convert it again.
const recent = new Map();

export async function readPage(startUrl, offset = 0) {
  let page = recent.get(startUrl);
  if (!page) {
    page = await fetchText(startUrl);
    recent.set(startUrl, page);
    if (recent.size > RECENT_PAGES) recent.delete(recent.keys().next().value);
  }
  return `Title: ${page.title}\nURL: ${page.url}\n\n${sliceText(page.text, offset)}`;
}
