import type { Stage } from "@/lib/api";

const LABELS: Record<Stage, string> = {
  searching: "searching",
  grading: "grading",
  rewriting: "rewriting query",
  answering: "answering",
};

const ORDER: Stage[] = ["searching", "grading", "answering"];

export function StatusIndicator({ stage, query }: { stage: Stage; query?: string }) {
  const activeIndex = stage === "rewriting" ? 0 : ORDER.indexOf(stage);
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-muted" aria-live="polite">
      {ORDER.map((step, index) => (
        <span
          key={step}
          className={`rounded-full border px-2 py-0.5 ${
            index === activeIndex
              ? "border-accent text-accent"
              : index < activeIndex
                ? "border-border text-muted line-through"
                : "border-border"
          }`}
        >
          {index === activeIndex ? LABELS[stage] : LABELS[step]}
        </span>
      ))}
      {query && stage === "searching" && <span className="truncate">for &ldquo;{query}&rdquo;</span>}
    </div>
  );
}
