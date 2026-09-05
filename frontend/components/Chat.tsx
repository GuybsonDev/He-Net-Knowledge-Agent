"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

import { Sources } from "@/components/Sources";
import { StatusIndicator } from "@/components/StatusIndicator";
import { ThemeToggle } from "@/components/ThemeToggle";
import { API_URL, DoneEvent, Source, Stage, askStream } from "@/lib/api";

type Turn = {
  id: number;
  question: string;
  answer: string;
  sources: Source[];
  stage: Stage | null;
  query?: string;
  usage?: Pick<DoneEvent, "input_tokens" | "output_tokens" | "cost_usd" | "model" | "rewrites">;
  error?: string;
};

export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const totals = turns.reduce(
    (sum, turn) => ({
      tokens: sum.tokens + (turn.usage ? turn.usage.input_tokens + turn.usage.output_tokens : 0),
      cost: sum.cost + (turn.usage?.cost_usd ?? 0),
    }),
    { tokens: 0, cost: 0 },
  );

  function patch(id: number, update: (turn: Turn) => Turn) {
    setTurns((current) => current.map((turn) => (turn.id === id ? update(turn) : turn)));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;

    const id = Date.now();
    setTurns((current) => [...current, { id, question: text, answer: "", sources: [], stage: null }]);
    setQuestion("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await askStream(
        text,
        threadId,
        {
          onStart: (thread) => setThreadId(thread),
          onStatus: (stage, query) => patch(id, (turn) => ({ ...turn, stage, query })),
          onDelta: (delta) => patch(id, (turn) => ({ ...turn, answer: turn.answer + delta })),
          onSources: (sources) => patch(id, (turn) => ({ ...turn, sources })),
          onDone: (done) =>
            patch(id, (turn) => ({
              ...turn,
              stage: null,
              answer: turn.answer || done.answer,
              sources: done.sources,
              usage: {
                input_tokens: done.input_tokens,
                output_tokens: done.output_tokens,
                cost_usd: done.cost_usd,
                model: done.model,
                rewrites: done.rewrites,
              },
            })),
          onError: (message) => patch(id, (turn) => ({ ...turn, stage: null, error: message })),
        },
        controller.signal,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      patch(id, (turn) => ({ ...turn, stage: null, error: `Could not reach ${API_URL}: ${message}` }));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 py-6">
      <header className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">He-Net Answers</h1>
          <p className="text-sm text-muted">
            Answers come only from henet.com.br. Every reply lists the pages it used.
          </p>
        </div>
        <ThemeToggle />
      </header>

      <section className="flex-1 space-y-4">
        {turns.length === 0 && (
          <p className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted">
            Try: &ldquo;Quais planos de internet a He-Net oferece?&rdquo;
          </p>
        )}
        {turns.map((turn) => (
          <article key={turn.id} className="space-y-2">
            <p className="ml-auto w-fit max-w-[85%] rounded-2xl bg-accent px-4 py-2 text-sm text-white">
              {turn.question}
            </p>
            <div className="rounded-2xl border border-border bg-panel px-4 py-3 text-sm">
              {turn.stage && <StatusIndicator stage={turn.stage} query={turn.query} />}
              {turn.answer && <p className="mt-2 whitespace-pre-wrap leading-relaxed">{turn.answer}</p>}
              {turn.error && <p className="mt-2 text-red-600 dark:text-red-400">{turn.error}</p>}
              <Sources sources={turn.sources} />
              {turn.usage && (
                <p className="mt-3 font-mono text-xs text-muted">
                  {turn.usage.model} · {turn.usage.input_tokens} in / {turn.usage.output_tokens} out ·
                  USD {turn.usage.cost_usd.toFixed(5)}
                  {turn.usage.rewrites > 0 && ` · ${turn.usage.rewrites} rewrite(s)`}
                </p>
              )}
            </div>
          </article>
        ))}
        <div ref={bottomRef} />
      </section>

      <form onSubmit={submit} className="sticky bottom-0 mt-6 bg-background pb-2 pt-3">
        <div className="flex gap-2">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about plans, coverage or support"
            aria-label="Question"
            disabled={busy}
            className="flex-1 rounded-lg border border-border bg-panel px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={busy || !question.trim()}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {busy ? "..." : "Ask"}
          </button>
        </div>
        <p className="mt-2 flex justify-between font-mono text-xs text-muted">
          <span>{turns.length} question(s) this session</span>
          <span>
            {totals.tokens} tokens · USD {totals.cost.toFixed(5)}
          </span>
        </p>
      </form>
    </main>
  );
}
