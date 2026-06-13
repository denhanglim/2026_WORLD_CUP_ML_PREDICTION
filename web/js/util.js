// Pure helpers for THE ALBUM '26 dashboard.
// No DOM, no side effects — unit-tested in web/tests/util.test.html.

/** Format a probability (0..1) as a percentage string, e.g. 0.2034 -> "20.3%". */
export function pct(p, digits = 1) {
  if (typeof p !== "number" || !isFinite(p)) return "0%";
  const v = p * 100;
  // Keep "0%" clean for true zeros; otherwise fixed digits.
  if (v === 0) return "0%";
  return `${v.toFixed(digits)}%`;
}

/** Bar width as a 0..100 number, scaled against the field leader so bars are legible. */
export function barWidth(p, maxP) {
  if (typeof p !== "number" || !isFinite(p) || p <= 0) return 0;
  if (typeof maxP !== "number" || maxP <= 0) return 0;
  const w = (p / maxP) * 100;
  return Math.max(2, Math.min(100, w)); // floor at 2% so tiny values still show a sliver
}

/** True if a rank should get the rare gold-foil treatment (top N contenders). */
export function isFoil(rank, foilCount = 6) {
  return typeof rank === "number" && rank >= 1 && rank <= foilCount;
}

/** Rotating pastel tint class for non-foil stickers, stable per rank. */
const TINTS = ["sky", "mint", "peach"];
export function tintFor(rank) {
  if (typeof rank !== "number" || rank < 1) return TINTS[0];
  return TINTS[(rank - 1) % TINTS.length];
}

/** "GRP A" group chip label. */
export function groupChip(group) {
  return `GRP ${String(group || "").toUpperCase()}`;
}

/**
 * The ladder rounds for a team, deepest-first (Qualify -> Win), each with label + prob.
 * Returns [] for a missing team.
 */
export function ladder(team) {
  if (!team) return [];
  return [
    { key: "p_qualify", label: "Qualify", p: team.p_qualify },
    { key: "p_r16", label: "Round of 16", p: team.p_r16 },
    { key: "p_qf", label: "Quarter-final", p: team.p_qf },
    { key: "p_sf", label: "Semi-final", p: team.p_sf },
    { key: "p_final", label: "Final", p: team.p_final },
    { key: "p_win", label: "Win it", p: team.p_win },
  ];
}

/** Safe accessor: max p_win across the field (for bar scaling). */
export function maxWin(teams) {
  if (!Array.isArray(teams) || teams.length === 0) return 1;
  return teams.reduce((m, t) => Math.max(m, t.p_win || 0), 0) || 1;
}
