"use client";

import { useSyncExternalStore } from "react";

function subscribe(callback: () => void): () => void {
  const observer = new MutationObserver(callback);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  return () => observer.disconnect();
}

function readDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

export function ThemeToggle() {
  // The server never knows the theme, so it renders light and the client corrects it.
  const dark = useSyncExternalStore(subscribe, readDark, () => false);

  function toggle() {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      // Private mode may block storage. The toggle still works for the session.
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="rounded-md border border-border px-3 py-1.5 text-sm text-muted hover:text-foreground"
    >
      {dark ? "Light" : "Dark"}
    </button>
  );
}
