import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { unansweredBy } from "./search.mjs";
import {
  assertPublicUrl,
  documentKind,
  docxToText,
  htmlToText,
  pdfToText,
  isPrivateAddress,
  PAGE_CHARS,
  publicLookup,
  readPage,
  sliceText,
} from "./page.mjs";

test("private and loopback addresses are recognized", () => {
  for (const address of ["127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1"]) {
    assert.equal(isPrivateAddress(address), true, address);
  }
  for (const address of ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"]) {
    assert.equal(isPrivateAddress(address), false, address);
  }
});

test("local and non-web URLs are refused", async () => {
  for (const url of ["http://127.0.0.1:8888/", "http://localhost:11434/", "http://[::1]/", "http://169.254.169.254/latest", "file:///C:/secrets.txt"]) {
    await assert.rejects(assertPublicUrl(new URL(url)), undefined, url);
  }
});

test("a name that resolves to a private address is refused when connecting", async () => {
  const resolve = (host, options) =>
    new Promise((done) => publicLookup(host, options, (error, result) => done({ error, result })));
  for (const options of [{}, { all: true }]) {
    const { error } = await resolve("localhost", options);
    assert.match(error.message, /private network address/);
  }
  await assert.rejects(readPage("http://localhost:11434/"), /private network address/);
});

test("html is reduced to the article text", () => {
  const paragraph = "<p>The actual article body is long enough to count as the main content of this page. </p>".repeat(10);
  const html = `<html><head><title>My title</title><script>alert(1)</script></head>
    <body><nav><a href="/">Home</a><a href="/about">About</a></nav><article><h1>Heading</h1>${paragraph}</article></body></html>`;
  const { title, text } = htmlToText(html, "https://example.com/post");
  assert.equal(title, "My title");
  assert.match(text, /actual article body/);
  assert.doesNotMatch(text, /alert\(1\)/);
});

test("a page that is mostly lists keeps its lists", () => {
  const rows = Array.from({ length: 60 }, (_, i) => `<tr><td><a href="/r/${i}">Release 3.${i}.0</a></td><td>2026-01-${i}</td></tr>`).join("");
  const aside = "<p>Release files are signed, and this long paragraph explains how to verify a download in careful detail. </p>".repeat(8);
  const html = `<html><head><title>Downloads</title></head><body><nav><a href="/">Home</a></nav>
    <main><h1>Download the latest version</h1><p><a href="/latest">Download Version 9.8.7</a></p>
    <table>${rows}</table><div class="article"><h2>Verifying</h2>${aside}</div></main>
    <footer>Copyright notice</footer></body></html>`;
  const { text } = htmlToText(html, "https://example.com/downloads");
  assert.match(text, /Version 9\.8\.7/);
  assert.match(text, /Release 3\.59\.0/);
  assert.doesNotMatch(text, /Copyright notice|Home/);
});

test("pdf and word documents are read as text", async () => {
  const pdf = await pdfToText(readFileSync(new URL("./fixtures/kestrel-contract.pdf", import.meta.url)));
  assert.match(pdf.text, /^\[Page 1\]\nKestrel maintenance contract/);
  assert.match(pdf.text, /\[Page 2\]\n[^]*Ilse Brandt/);
  const docx = await docxToText(readFileSync(new URL("./fixtures/osprey-onboarding.docx", import.meta.url)));
  assert.match(docx.text, /laptop of model TX-880/);
});

test("documents are recognized by type, or by name when sent as a plain download", () => {
  assert.equal(documentKind("application/pdf; charset=binary", "/a"), "pdf");
  assert.equal(documentKind("application/octet-stream", "/files/Report.PDF"), "pdf");
  assert.equal(documentKind("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "/a"), "docx");
  assert.equal(documentKind("", "/notes.docx"), "docx");
  assert.equal(documentKind("text/html", "/looks-like.pdf"), null);
  assert.equal(documentKind("application/zip", "/archive.zip"), null);
});

test("a search every engine refused is an error, not an empty result", () => {
  const refused = { results: [], unresponsive_engines: [["brave", "Suspended: too many requests"], ["duckduckgo", "CAPTCHA"]] };
  assert.match(unansweredBy(refused), /^Web search is unavailable: .*brave: Suspended.*duckduckgo: CAPTCHA/);
  assert.equal(unansweredBy({ results: [], unresponsive_engines: [] }), null, "a real empty result");
  assert.equal(unansweredBy({ results: [{ url: "https://a.org" }], unresponsive_engines: [["brave", "timeout"]] }), null);
});

test("long text is split with a continuation note", () => {
  const text = "x".repeat(PAGE_CHARS + 100);
  assert.match(sliceText(text, 0), new RegExp(`offset=${PAGE_CHARS}\\.\\]$`));
  assert.equal(sliceText(text, PAGE_CHARS), "x".repeat(100));
});
