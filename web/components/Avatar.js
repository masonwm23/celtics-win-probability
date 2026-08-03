"use client";

import { useState } from "react";
import { initials } from "@/lib/format";

/**
 * A player headshot that degrades honestly.
 *
 * Roughly one percent of playing time belongs to players whose CDN photo may
 * not exist. When the image fails we show their initials rather than a generic
 * silhouette, because a silhouette reads as "this player" and initials read as
 * "no photo", which is the true statement.
 */
export default function Avatar({ player, size = "" }) {
  const [failed, setFailed] = useState(false);
  const className = `avatar ${size === "sm" ? "avatar--sm" : ""}`;

  if (failed || !player?.headshot) {
    return (
      <div className={`${className} avatar--fallback`} aria-hidden="true">
        {initials(player?.name)}
      </div>
    );
  }
  return (
    <img
      className={className}
      src={player.headshot}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}
