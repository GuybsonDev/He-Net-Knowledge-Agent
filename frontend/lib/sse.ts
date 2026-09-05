export type SSEEvent = { event: string; data: unknown };

// Incremental SSE parser. Fetch chunks do not line up with event boundaries, so
// leftover text is kept between feed() calls.
export class SSEParser {
  private buffer = "";

  feed(chunk: string): SSEEvent[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const events: SSEEvent[] = [];
    let separator = this.buffer.indexOf("\n\n");
    while (separator !== -1) {
      const block = this.buffer.slice(0, separator);
      this.buffer = this.buffer.slice(separator + 2);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
      separator = this.buffer.indexOf("\n\n");
    }
    return events;
  }
}

function parseBlock(block: string): SSEEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { event, data: JSON.parse(dataLines.join("\n")) };
}
