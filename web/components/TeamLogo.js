"use client";

import { useState } from "react";

/**
 * A team logo that degrades to the tricode.
 *
 * The NBA CDN is reachable from a browser but not from every environment, and a
 * broken-image icon looks like a bug. The abbreviation is always correct, so it
 * is the right fallback.
 */
export default function TeamLogo({ src, abbr, size = 52, className = "ribbon__logo" }) {
  const [failed, setFailed] = useState(false);
  if (failed || !src) {
    return (
      <div
        style={{
          width: size,
          height: size,
          borderRadius: 12,
          display: "grid",
          placeItems: "center",
          border: "1px solid var(--line)",
          background: "var(--bg-raised)",
          color: "var(--text-faint)",
          fontWeight: 700,
          fontSize: size * 0.3,
          letterSpacing: 0.5,
        }}
        className={className}
        aria-hidden="true"
      >
        {abbr}
      </div>
    );
  }
  return (
    <img
      className={className}
      style={{ width: size, height: size }}
      src={src}
      alt=""
      onError={() => setFailed(true)}
    />
  );
}
