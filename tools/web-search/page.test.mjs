import assert from "node:assert/strict";
import { test } from "node:test";
import { assertPublicUrl, htmlToText, isPrivateAddress, PAGE_CHARS, sliceText } from "./page.mjs";

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

test("html is reduced to the article text", () => {
  const paragraph = "<p>The actual article body is long enough to count as the main content of this page. </p>".repeat(10);
  const html = `<html><head><title>My title</title><script>alert(1)</script></head>
    <body><nav><a href="/">Home</a><a href="/about">About</a></nav><article><h1>Heading</h1>${paragraph}</article></body></html>`;
  const { title, text } = htmlToText(html, "https://example.com/post");
  assert.equal(title, "My title");
  assert.match(text, /actual article body/);
  assert.doesNotMatch(text, /alert\(1\)/);
});

test("long text is split with a continuation note", () => {
  const text = "x".repeat(PAGE_CHARS + 100);
  assert.match(sliceText(text, 0), new RegExp(`offset=${PAGE_CHARS}\\.\\]$`));
  assert.equal(sliceText(text, PAGE_CHARS), "x".repeat(100));
});
