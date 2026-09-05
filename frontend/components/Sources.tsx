import type { Source } from "@/lib/api";

export function Sources({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-border pt-2">
      <p className="mb-1 text-xs uppercase tracking-wide text-muted">Sources</p>
      <ol className="space-y-1 text-sm">
        {sources.map((source, index) => (
          <li key={source.url} className="flex gap-2">
            <span className="text-muted">{index + 1}.</span>
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer"
              className="text-accent underline-offset-2 hover:underline"
            >
              {source.title || source.url}
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}
