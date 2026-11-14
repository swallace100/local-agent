const BLOCKED_PREFIX = "Web search is unavailable:";

// Search engines block a client that asks too often. That must not look like "nothing
// found", or a report would say there was no evidence when really no search ran.
export function unansweredBy(body) {
  const failed = body.unresponsive_engines ?? [];
  if ((body.results ?? []).length || !failed.length) return null;
  const names = failed.map(([name, reason]) => `${name}: ${reason}`).join(", ");
  return (
    `${BLOCKED_PREFIX} every search engine refused the request (${names}). This usually ` +
    "clears within an hour. Searching again now won't help, so use what you have and say " +
    "that web search was unavailable."
  );
}
