import { SSEParser } from "@/lib/sse";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Stage = "searching" | "grading" | "rewriting" | "answering";

export type Source = { url: string; title: string };

export type DoneEvent = {
  answer: string;
  sources: Source[];
  query: string;
  grade: string;
  rewrites: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  thread_id: string;
  model: string;
  tool_rounds: number;
};

export type StreamHandlers = {
  onStart?: (threadId: string) => void;
  onStatus?: (stage: Stage, query?: string) => void;
  onDelta?: (text: string) => void;
  onSources?: (sources: Source[]) => void;
  onDone?: (done: DoneEvent) => void;
  onError?: (message: string) => void;
};

export async function askStream(
  question: string,
  threadId: string | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, thread_id: threadId }),
    signal,
  });
  if (!response.ok || !response.body) {
    handlers.onError?.(`API returned ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const { event, data } of parser.feed(decoder.decode(value, { stream: true }))) {
      dispatch(event, data, handlers);
    }
  }
}

function dispatch(event: string, data: unknown, handlers: StreamHandlers): void {
  const payload = data as Record<string, unknown>;
  switch (event) {
    case "start":
      handlers.onStart?.(String(payload.thread_id));
      break;
    case "status":
      handlers.onStatus?.(payload.stage as Stage, payload.query as string | undefined);
      break;
    case "delta":
      handlers.onDelta?.(String(payload.text ?? ""));
      break;
    case "sources":
      handlers.onSources?.((payload.sources as Source[]) ?? []);
      break;
    case "done":
      handlers.onDone?.(payload as unknown as DoneEvent);
      break;
    case "error":
      handlers.onError?.(String(payload.message ?? "unknown error"));
      break;
  }
}
