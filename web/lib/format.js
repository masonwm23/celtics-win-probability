/** Small formatting helpers, kept in one place so the components stay readable. */

export function formatClock(raw) {
  // The feed gives ISO 8601 durations with hundredths: PT06M47.00S.
  if (typeof raw !== "string") return "--:--";
  const match = raw.match(/PT(\d+)M([\d.]+)S/);
  if (!match) return "--:--";
  const minutes = Number(match[1]);
  const seconds = Math.floor(Number(match[2]));
  // Under a minute the tenths matter, which is how a broadcast shows it.
  if (minutes === 0) {
    const tenths = Math.floor((Number(match[2]) % 1) * 10);
    return `${seconds}.${tenths}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function periodLabel(period) {
  if (period <= 4) return `Q${period}`;
  return period === 5 ? "OT" : `${period - 4}OT`;
}

export const percent = (p, digits = 1) => `${(p * 100).toFixed(digits)}%`;

export function signed(value, digits = 0) {
  const rounded = Number(value).toFixed(digits);
  return Number(value) > 0 ? `+${rounded}` : rounded;
}

export function initials(name) {
  if (!name) return "?";
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

/** A short, readable date: "8 Nov 2018". */
export function prettyDate(iso) {
  if (!iso) return "";
  const [y, m, d] = iso.split("-").map(Number);
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${d} ${months[m - 1]} ${y}`;
}

/**
 * A player's shorthand for the court: jersey if there is one, else initials.
 */
export function courtLabel(player) {
  if (!player) return "";
  if (player.jersey) return jerseyNumber(player.jersey);
  return initials(player.name);
}

/**
 * Jersey numbers arrive from the serving layer as "7.0", "0.0", "50.0": a
 * float that was stringified somewhere upstream in build_serving.py. Trimming
 * the decimal here keeps the validated data files untouched, at the cost of
 * this note explaining why the display is doing arithmetic on a label.
 */
export function jerseyNumber(raw) {
  const text = String(raw).trim();
  const match = text.match(/^(\d+)(?:\.0+)?$/);
  return match ? match[1] : text;
}

export const POSITION_GROUPS = [
  { key: "G", label: "Guards", match: (p) => (p || "").startsWith("G") },
  { key: "F", label: "Forwards", match: (p) => (p || "").startsWith("F") },
  { key: "C", label: "Centers", match: (p) => (p || "").startsWith("C") },
];

export function positionGroup(player) {
  const value = player.position || player.coarse_position || "";
  const group = POSITION_GROUPS.find((g) => g.match(value));
  return group ? group.key : "?";
}


/**
 * The name to put under a marker.
 *
 * Not simply the last token: "Michael Porter Jr." would display as "Jr." and
 * "Larry Nance Jr." as the same thing, which is how two different players end
 * up looking identical on a court. Generational suffixes are carried along
 * with the surname instead.
 */
const SUFFIXES = new Set(["jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"]);

export function displaySurname(name) {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "";
  const last = parts[parts.length - 1];
  if (parts.length > 1 && SUFFIXES.has(last.toLowerCase())) {
    return `${parts[parts.length - 2]} ${last}`;
  }
  return last;
}
