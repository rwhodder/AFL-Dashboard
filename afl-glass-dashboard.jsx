import { useState, useMemo, useCallback, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from "recharts";

/* ═══════════════════════════════════════════════════════════════
   DARK GLASS — Premium analytics terminal aesthetic
   
   Ref: Dark fintech dashboards, Stripe Radar, Linear app,
   high-end analytics panels with luminous data on black glass.
   
   Principles:
   - Depth from layered glass surfaces (not borders/brackets)
   - Light comes FROM the data (glowing numbers, chart fills)
   - Minimal chrome — let typography and colour do the work
   - Every glow is functional: green=profit, red=loss, blue=info
═══════════════════════════════════════════════════════════════ */

const C = {
  bg: "#09090b",
  surface: "#111114",
  glass: "#16161a",
  glassBorder: "#222228",
  glassHover: "#1c1c22",
  text: "#e4e4e7",
  textSoft: "#a1a1aa",
  textMuted: "#52525b",
  textDim: "#2e2e35",
  green: "#34d399",
  greenSoft: "#34d39920",
  greenGlow: "#34d39940",
  red: "#f87171",
  redSoft: "#f8717120",
  amber: "#fbbf24",
  amberSoft: "#fbbf2420",
  cyan: "#22d3ee",
  s1: "#34d399",
  s2: "#60a5fa",
  s3: "#a78bfa",
  gold: "#fbbf24",
};

const font = {
  mono: "'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', monospace",
  sans: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  num: "'SF Mono', 'Tabular Nums', monospace",
};

const glass = (extra = {}) => ({
  background: C.glass,
  border: `1px solid ${C.glassBorder}`,
  borderRadius: 10,
  ...extra,
});

const injectOnce = () => {
  if (document.getElementById("dg-styles")) return;
  const s = document.createElement("style");
  s.id = "dg-styles";
  s.textContent = `
    .dg-row { transition: background 0.15s; }
    .dg-row:hover { background: ${C.glassHover} !important; }
    .dg-btn { transition: all 0.15s; }
    .dg-btn:hover { background: #ffffff0a !important; border-color: #ffffff18 !important; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
  `;
  document.head.appendChild(s);
};

// ─── Ring Gauge ──────────────────────────────────────────────────
const Ring = ({ value, max = 100, color, size = 100, label }) => {
  const t = 4;
  const r = (size - t * 2) / 2;
  const circ = 2 * Math.PI * r;
  const prog = (value / max) * circ;
  const cx = size / 2, cy = size / 2;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color + "12"} strokeWidth={t} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={t}
          strokeDasharray={`${prog} ${circ}`} strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 8px ${color}60)`, transition: "stroke-dasharray 0.6s ease" }} />
      </svg>
      <div style={{
        position: "absolute", inset: 0, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
      }}>
        <span style={{
          fontFamily: font.num, fontSize: size * 0.24, fontWeight: 700, color,
          lineHeight: 1, letterSpacing: "-0.02em",
        }}>{value.toFixed(1)}</span>
        {label && <span style={{ fontFamily: font.mono, fontSize: 8, color: C.textMuted, letterSpacing: "0.1em", marginTop: 3 }}>{label}</span>}
      </div>
    </div>
  );
};

// ─── Pill ────────────────────────────────────────────────────────
const Pill = ({ children, color }) => (
  <span style={{
    display: "inline-block", padding: "1px 6px", fontSize: 9,
    fontFamily: font.mono, fontWeight: 600, letterSpacing: "0.05em",
    background: color + "14", color, borderRadius: 3,
  }}>{children}</span>
);

// ─── MOCK DATA ───────────────────────────────────────────────────
const ROUND = 4;
const modelOutput = [
  { id: 1, player: "N. Daicos", team: "COL", opp: "ESS", line: 28.5, avg: 31.2, last3: 33.0, last5: 30.8, dvp: 1.08, consistency: 92, weather: "Clear", ground: "MCG", strategy: "S1", priority: 1, prediction: 32.1, edge: 3.6, conf: "HIGH" },
  { id: 2, player: "M. Gawn", team: "MEL", opp: "GWS", line: 34.5, avg: 36.8, last3: 38.2, last5: 35.9, dvp: 1.12, consistency: 85, weather: "Clear", ground: "MCG", strategy: "S1", priority: 2, prediction: 37.4, edge: 2.9, conf: "HIGH" },
  { id: 3, player: "T. Miller", team: "GCS", opp: "WCE", line: 27.5, avg: 29.1, last3: 26.8, last5: 28.4, dvp: 1.15, consistency: 78, weather: "Cloudy", ground: "PCS", strategy: "S3", priority: 5, prediction: 29.8, edge: 2.3, conf: "MED" },
  { id: 4, player: "L. Neale", team: "BRL", opp: "SYD", line: 30.5, avg: 33.4, last3: 35.1, last5: 32.7, dvp: 0.98, consistency: 88, weather: "Rain", ground: "GAB", strategy: "S1", priority: 3, prediction: 32.0, edge: 1.5, conf: "MED" },
  { id: 5, player: "C. Oliver", team: "MEL", opp: "GWS", line: 29.5, avg: 31.0, last3: 28.5, last5: 30.2, dvp: 1.05, consistency: 82, weather: "Clear", ground: "MCG", strategy: "S3", priority: 6, prediction: 30.9, edge: 1.4, conf: "MED" },
  { id: 6, player: "Z. Merrett", team: "ESS", opp: "COL", line: 26.5, avg: 28.9, last3: 30.2, last5: 29.1, dvp: 1.02, consistency: 80, weather: "Clear", ground: "MCG", strategy: "S1", priority: 4, prediction: 29.3, edge: 2.8, conf: "HIGH" },
  { id: 7, player: "J. Macrae", team: "WBD", opp: "HAW", line: 25.5, avg: 26.1, last3: 24.8, last5: 25.9, dvp: 0.95, consistency: 71, weather: "Clear", ground: "MRV", strategy: "S3", priority: 8, prediction: 25.8, edge: 0.3, conf: "LOW" },
  { id: 8, player: "T. Green", team: "GWS", opp: "MEL", line: 24.5, avg: 27.2, last3: 28.9, last5: 26.8, dvp: 1.01, consistency: 76, weather: "Clear", ground: "MCG", strategy: "S1", priority: 7, prediction: 27.0, edge: 2.5, conf: "MED" },
  { id: 9, player: "J. Dunkley", team: "BRL", opp: "SYD", line: 23.5, avg: 25.8, last3: 27.1, last5: 25.2, dvp: 0.98, consistency: 74, weather: "Rain", ground: "GAB", strategy: "S3", priority: 9, prediction: 24.9, edge: 1.4, conf: "MED" },
  { id: 10, player: "A. Cerra", team: "CAR", opp: "NME", line: 22.5, avg: 24.1, last3: 25.8, last5: 23.9, dvp: 1.09, consistency: 69, weather: "Clear", ground: "MRV", strategy: "S3", priority: 10, prediction: 24.5, edge: 2.0, conf: "MED" },
  { id: 11, player: "E. Yeo", team: "WCE", opp: "GCS", line: 21.5, avg: 20.8, last3: 19.2, last5: 20.5, dvp: 0.92, consistency: 65, weather: "Cloudy", ground: "PCS", strategy: null, priority: null, prediction: 20.1, edge: -1.4, conf: "LOW" },
  { id: 12, player: "S. Ross", team: "STK", opp: "FRE", line: 20.5, avg: 19.2, last3: 17.8, last5: 19.0, dvp: 0.88, consistency: 58, weather: "Clear", ground: "MRV", strategy: null, priority: null, prediction: 18.5, edge: -2.0, conf: "LOW" },
];

const performanceData = {
  S1: { name: "High Confidence", colour: C.s1, winRate: 81.2, pushRate: 4.3, lossRate: 14.5, totalBets: 69, threshold: "Edge ≥ 2.0 · Con ≥ 80", roi: 12.4, streak: "+5W", status: "ACTIVE", rounds: [
    { round: 1, wins: 5, losses: 1, pushes: 0, total: 6, winPct: 83.3, cumWinPct: 83.3, profit: 3.2 },
    { round: 2, wins: 4, losses: 1, pushes: 1, total: 6, winPct: 66.7, cumWinPct: 75.0, profit: 5.1 },
    { round: 3, wins: 6, losses: 1, pushes: 0, total: 7, winPct: 85.7, cumWinPct: 78.9, profit: 9.4 },
    { round: 4, wins: 5, losses: 0, pushes: 0, total: 5, winPct: 100, cumWinPct: 83.3, profit: 14.1 },
    { round: 5, wins: 4, losses: 1, pushes: 0, total: 5, winPct: 80.0, cumWinPct: 82.8, profit: 16.8 },
    { round: 6, wins: 5, losses: 1, pushes: 1, total: 7, winPct: 71.4, cumWinPct: 80.6, profit: 18.2 },
    { round: 7, wins: 6, losses: 0, pushes: 0, total: 6, winPct: 100, cumWinPct: 83.3, profit: 23.8 },
    { round: 8, wins: 4, losses: 2, pushes: 0, total: 6, winPct: 66.7, cumWinPct: 81.3, profit: 24.1 },
    { round: 9, wins: 5, losses: 0, pushes: 1, total: 6, winPct: 83.3, cumWinPct: 81.5, profit: 28.5 },
    { round: 10, wins: 4, losses: 1, pushes: 0, total: 5, winPct: 80.0, cumWinPct: 81.4, profit: 30.2 },
    { round: 11, wins: 6, losses: 1, pushes: 0, total: 7, winPct: 85.7, cumWinPct: 81.9, profit: 34.8 },
    { round: 12, wins: 5, losses: 1, pushes: 0, total: 6, winPct: 83.3, cumWinPct: 82.1, profit: 37.5 },
  ]},
  S2: { name: "Weather Exploits", colour: C.s2, winRate: 62.5, pushRate: 6.3, lossRate: 31.2, totalBets: 32, threshold: "Rain + DvP ≥ 1.05", roi: -3.2, streak: "-2L", status: "BENCHED", rounds: [
    { round: 1, wins: 2, losses: 1, pushes: 0, total: 3, winPct: 66.7, cumWinPct: 66.7, profit: 0.8 },
    { round: 2, wins: 1, losses: 1, pushes: 1, total: 3, winPct: 33.3, cumWinPct: 50.0, profit: -0.5 },
    { round: 3, wins: 2, losses: 0, pushes: 0, total: 2, winPct: 100, cumWinPct: 62.5, profit: 1.4 },
    { round: 4, wins: 1, losses: 2, pushes: 0, total: 3, winPct: 33.3, cumWinPct: 54.5, profit: -1.2 },
    { round: 5, wins: 2, losses: 1, pushes: 0, total: 3, winPct: 66.7, cumWinPct: 57.1, profit: -0.1 },
    { round: 6, wins: 1, losses: 1, pushes: 1, total: 3, winPct: 33.3, cumWinPct: 52.9, profit: -1.5 },
    { round: 7, wins: 2, losses: 1, pushes: 0, total: 3, winPct: 66.7, cumWinPct: 55.0, profit: -0.4 },
    { round: 8, wins: 1, losses: 2, pushes: 0, total: 3, winPct: 33.3, cumWinPct: 52.0, profit: -2.8 },
    { round: 9, wins: 2, losses: 0, pushes: 1, total: 3, winPct: 66.7, cumWinPct: 53.8, profit: -1.5 },
    { round: 10, wins: 1, losses: 1, pushes: 0, total: 2, winPct: 50.0, cumWinPct: 53.3, profit: -2.1 },
    { round: 11, wins: 2, losses: 1, pushes: 0, total: 3, winPct: 66.7, cumWinPct: 54.5, profit: -1.0 },
    { round: 12, wins: 0, losses: 2, pushes: 0, total: 2, winPct: 0, cumWinPct: 51.5, profit: -3.2 },
  ]},
  S3: { name: "Value Unders", colour: C.s3, winRate: 69.4, pushRate: 3.2, lossRate: 27.4, totalBets: 62, threshold: "Edge ≥ 1.0 · DvP ≥ 0.95", roi: 5.8, streak: "+3W", status: "ACTIVE", rounds: [
    { round: 1, wins: 3, losses: 2, pushes: 0, total: 5, winPct: 60.0, cumWinPct: 60.0, profit: 0.5 },
    { round: 2, wins: 4, losses: 1, pushes: 1, total: 6, winPct: 66.7, cumWinPct: 63.6, profit: 2.8 },
    { round: 3, wins: 3, losses: 2, pushes: 0, total: 5, winPct: 60.0, cumWinPct: 62.5, profit: 3.2 },
    { round: 4, wins: 4, losses: 1, pushes: 0, total: 5, winPct: 80.0, cumWinPct: 66.7, profit: 5.8 },
    { round: 5, wins: 3, losses: 2, pushes: 0, total: 5, winPct: 60.0, cumWinPct: 65.4, profit: 6.1 },
    { round: 6, wins: 4, losses: 1, pushes: 0, total: 5, winPct: 80.0, cumWinPct: 67.7, profit: 8.5 },
    { round: 7, wins: 3, losses: 2, pushes: 1, total: 6, winPct: 50.0, cumWinPct: 64.9, profit: 8.0 },
    { round: 8, wins: 4, losses: 1, pushes: 0, total: 5, winPct: 80.0, cumWinPct: 66.7, profit: 10.4 },
    { round: 9, wins: 3, losses: 2, pushes: 0, total: 5, winPct: 60.0, cumWinPct: 66.0, profit: 10.8 },
    { round: 10, wins: 5, losses: 1, pushes: 0, total: 6, winPct: 83.3, cumWinPct: 68.1, profit: 14.2 },
    { round: 11, wins: 3, losses: 2, pushes: 0, total: 5, winPct: 60.0, cumWinPct: 67.3, profit: 14.5 },
    { round: 12, wins: 4, losses: 1, pushes: 0, total: 5, winPct: 80.0, cumWinPct: 68.3, profit: 17.1 },
  ]},
};

// ─── MAIN ────────────────────────────────────────────────────────
export default function Dashboard() {
  useEffect(() => { injectOnce(); }, []);

  const [tab, setTab] = useState("picks");
  const [filterStrategy, setFilterStrategy] = useState("ALL");
  const [filterConf, setFilterConf] = useState("ALL");
  const [selectedBets, setSelectedBets] = useState(new Set());
  const [sortCol, setSortCol] = useState("priority");
  const [sortDir, setSortDir] = useState("asc");

  const qualified = useMemo(() => modelOutput.filter(b => b.strategy !== null), []);
  const rejected = useMemo(() => modelOutput.filter(b => b.strategy === null), []);

  const filtered = useMemo(() => {
    let b = [...qualified];
    if (filterStrategy !== "ALL") b = b.filter(x => x.strategy === filterStrategy);
    if (filterConf !== "ALL") b = b.filter(x => x.conf === filterConf);
    b.sort((a, c) => {
      const aV = a[sortCol], cV = c[sortCol];
      if (aV == null) return 1; if (cV == null) return -1;
      const d = typeof aV === "string" ? aV.localeCompare(cV) : aV - cV;
      return sortDir === "asc" ? d : -d;
    });
    return b;
  }, [qualified, filterStrategy, filterConf, sortCol, sortDir]);

  const toggle = useCallback((id) => {
    setSelectedBets(p => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }, []);

  const sort = useCallback((col) => {
    setSortCol(p => { if (p === col) { setSortDir(d => d === "asc" ? "desc" : "asc"); return col; } setSortDir("asc"); return col; });
  }, []);

  const selected = modelOutput.filter(b => selectedBets.has(b.id)).sort((a, b) => a.priority - b.priority);
  const pairs = useMemo(() => {
    const p = [];
    for (let i = 0; i < selected.length; i++)
      for (let j = i + 1; j < selected.length; j++)
        p.push([selected[i], selected[j]]);
    return p;
  }, [selectedBets]);

  const sc = s => s === "S1" ? C.s1 : s === "S2" ? C.s2 : s === "S3" ? C.s3 : C.textMuted;
  const cc = c => c === "HIGH" ? C.green : c === "MED" ? C.amber : C.textMuted;
  const ec = e => e >= 2.5 ? C.green : e >= 1.0 ? C.amber : C.red;

  // ─── PICKS TAB ─────────────────────────────────────────────────
  const Picks = () => (
    <div style={{ display: "flex", height: "calc(100vh - 52px)" }}>
      {/* Table */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Filters */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8, padding: "8px 16px",
          borderBottom: `1px solid ${C.glassBorder}`,
          fontFamily: font.mono, fontSize: 10,
        }}>
          <span style={{ color: C.textMuted, fontSize: 9, letterSpacing: "0.08em" }}>STRATEGY</span>
          {["ALL", "S1", "S2", "S3"].map(s => (
            <button key={s} className="dg-btn" onClick={() => setFilterStrategy(s)} style={{
              background: filterStrategy === s ? (s === "ALL" ? "#ffffff08" : sc(s) + "14") : "transparent",
              border: `1px solid ${filterStrategy === s ? (s === "ALL" ? "#ffffff15" : sc(s) + "40") : "transparent"}`,
              color: filterStrategy === s ? (s === "ALL" ? C.text : sc(s)) : C.textMuted,
              padding: "3px 10px", cursor: "pointer", fontFamily: font.mono, fontSize: 10,
              borderRadius: 5,
            }}>{s}</button>
          ))}
          <div style={{ width: 1, height: 14, background: C.glassBorder, margin: "0 4px" }} />
          <span style={{ color: C.textMuted, fontSize: 9, letterSpacing: "0.08em" }}>CONF</span>
          {["ALL", "HIGH", "MED", "LOW"].map(c => (
            <button key={c} className="dg-btn" onClick={() => setFilterConf(c)} style={{
              background: filterConf === c ? cc(c) + "14" : "transparent",
              border: `1px solid ${filterConf === c ? cc(c) + "40" : "transparent"}`,
              color: filterConf === c ? cc(c) : C.textMuted,
              padding: "3px 10px", cursor: "pointer", fontFamily: font.mono, fontSize: 10,
              borderRadius: 5,
            }}>{c}</button>
          ))}
          <span style={{ marginLeft: "auto", color: C.textMuted, fontSize: 10, fontFamily: font.mono }}>
            {filtered.length} <span style={{ color: C.textDim }}>of</span> {modelOutput.length}
          </span>
        </div>

        {/* Table body */}
        <div style={{ flex: 1, overflow: "auto", padding: "0 8px 8px" }}>
          <div style={{ ...glass({ padding: 0, overflow: "hidden", marginTop: 8 }) }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: font.mono, fontSize: 11 }}>
              <thead>
                <tr>
                  {[
                    { k: "_", l: "", w: 32, ns: 1 },
                    { k: "priority", l: "#", w: 32 },
                    { k: "player", l: "PLAYER", w: 120, left: 1 },
                    { k: "team", l: "TM", w: 36 },
                    { k: "opp", l: "OPP", w: 36 },
                    { k: "strategy", l: "STR", w: 36 },
                    { k: "line", l: "LINE", w: 48 },
                    { k: "prediction", l: "PRED", w: 48 },
                    { k: "edge", l: "EDGE", w: 50 },
                    { k: "avg", l: "AVG", w: 44 },
                    { k: "last3", l: "L3", w: 38 },
                    { k: "last5", l: "L5", w: 38 },
                    { k: "dvp", l: "DvP", w: 42 },
                    { k: "consistency", l: "CON", w: 38 },
                    { k: "conf", l: "CONF", w: 44 },
                    { k: "weather", l: "WX", w: 30 },
                    { k: "ground", l: "GND", w: 36 },
                  ].map(c => (
                    <th key={c.k} onClick={() => !c.ns && sort(c.k)} style={{
                      width: c.w, padding: "8px 5px", textAlign: c.left ? "left" : "right",
                      color: C.textMuted, fontWeight: 500, fontSize: 9,
                      letterSpacing: "0.06em", cursor: c.ns ? "default" : "pointer",
                      borderBottom: `1px solid ${C.glassBorder}`, userSelect: "none",
                      background: C.glass,
                    }}>
                      {c.l}
                      {sortCol === c.k && <span style={{ color: C.text, marginLeft: 2, fontSize: 8 }}>{sortDir === "asc" ? "↑" : "↓"}</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((b) => {
                  const sel = selectedBets.has(b.id);
                  return (
                    <tr key={b.id} className="dg-row" onClick={() => toggle(b.id)}
                      style={{
                        cursor: "pointer",
                        background: sel ? C.green + "08" : "transparent",
                        borderBottom: `1px solid ${C.glassBorder}50`,
                      }}>
                      <td style={{ padding: "5px 5px", textAlign: "center" }}>
                        <div style={{
                          width: 14, height: 14, borderRadius: 3,
                          border: `1.5px solid ${sel ? C.green : C.textDim}`,
                          background: sel ? C.green + "20" : "transparent",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          margin: "0 auto", transition: "all 0.15s",
                        }}>
                          {sel && <div style={{ width: 6, height: 6, borderRadius: 1, background: C.green }} />}
                        </div>
                      </td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: b.priority <= 3 ? C.gold : C.textMuted, fontWeight: b.priority <= 3 ? 700 : 400 }}>
                        {b.priority}
                      </td>
                      <td style={{ padding: "5px 5px", textAlign: "left", color: C.text, fontWeight: 600, fontFamily: font.sans, fontSize: 12 }}>
                        {b.player}
                      </td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: C.textSoft, fontSize: 10 }}>{b.team}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: C.textMuted, fontSize: 10 }}>{b.opp}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right" }}><Pill color={sc(b.strategy)}>{b.strategy}</Pill></td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: C.text, fontWeight: 600 }}>{b.line}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: b.prediction > b.line ? C.green : C.red, fontWeight: 700 }}>
                        {b.prediction}
                      </td>
                      <td style={{ padding: "5px 5px", textAlign: "right" }}>
                        <span style={{
                          color: ec(b.edge), fontWeight: 700,
                          background: ec(b.edge) + "10", padding: "1px 5px", borderRadius: 3,
                        }}>+{b.edge.toFixed(1)}</span>
                      </td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: C.textSoft }}>{b.avg}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: b.last3 > b.avg ? C.green : b.last3 < b.avg * 0.95 ? C.red : C.textSoft }}>{b.last3}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: C.textMuted }}>{b.last5}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: b.dvp >= 1.05 ? C.green : b.dvp <= 0.95 ? C.red : C.textSoft }}>{b.dvp.toFixed(2)}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: b.consistency >= 80 ? C.green : b.consistency >= 70 ? C.amber : C.red }}>{b.consistency}</td>
                      <td style={{ padding: "5px 5px", textAlign: "right" }}><Pill color={cc(b.conf)}>{b.conf}</Pill></td>
                      <td style={{ padding: "5px 5px", textAlign: "center", fontSize: 10, color: b.weather === "Rain" ? C.s2 : C.textDim }}>
                        {b.weather === "Rain" ? "●" : b.weather === "Cloudy" ? "◐" : "○"}
                      </td>
                      <td style={{ padding: "5px 5px", textAlign: "right", color: C.textDim, fontSize: 9 }}>{b.ground}</td>
                    </tr>
                  );
                })}
                {/* Rejected below line */}
                {filterStrategy === "ALL" && filterConf === "ALL" && rejected.map(b => (
                  <tr key={b.id} style={{ opacity: 0.28 }}>
                    <td style={{ padding: "5px 5px" }} />
                    <td style={{ padding: "5px 5px", textAlign: "right", color: C.textDim }}>—</td>
                    <td style={{ padding: "5px 5px", textAlign: "left", color: C.textMuted, fontFamily: font.sans, fontSize: 12 }}>{b.player}</td>
                    <td style={{ padding: "5px 5px", textAlign: "right", color: C.textDim, fontSize: 10 }}>{b.team}</td>
                    <td style={{ padding: "5px 5px", textAlign: "right", color: C.textDim, fontSize: 10 }}>{b.opp}</td>
                    <td style={{ padding: "5px 5px", textAlign: "right", color: C.textDim }}>—</td>
                    <td style={{ padding: "5px 5px", textAlign: "right", color: C.textDim }}>{b.line}</td>
                    <td style={{ padding: "5px 5px", textAlign: "right", color: C.red + "50" }}>{b.prediction}</td>
                    <td style={{ padding: "5px 5px", textAlign: "right", color: C.red + "50" }}>{b.edge.toFixed(1)}</td>
                    <td colSpan={8} />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div style={{
        width: 270, display: "flex", flexDirection: "column",
        borderLeft: `1px solid ${C.glassBorder}`,
        background: C.surface,
      }}>
        <div style={{
          padding: "10px 14px", borderBottom: `1px solid ${C.glassBorder}`,
          display: "flex", justifyContent: "space-between", alignItems: "baseline",
        }}>
          <span style={{ fontFamily: font.mono, fontSize: 10, color: C.textMuted, letterSpacing: "0.06em" }}>MULTI BUILDER</span>
          <span style={{ fontFamily: font.num, fontSize: 22, fontWeight: 700, color: C.text }}>{selected.length}</span>
        </div>

        <div style={{ padding: "10px 14px", borderBottom: `1px solid ${C.glassBorder}` }}>
          <div style={{ fontSize: 9, fontFamily: font.mono, color: C.textMuted, letterSpacing: "0.06em", marginBottom: 6 }}>SINGLES</div>
          {selected.length === 0 ? (
            <div style={{ fontSize: 10, fontFamily: font.mono, color: C.textDim, padding: "10px 0" }}>Select rows to build legs</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {selected.map(b => (
                <div key={b.id} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "4px 8px", borderRadius: 5,
                  background: sc(b.strategy) + "08",
                  borderLeft: `2px solid ${sc(b.strategy)}40`,
                  fontFamily: font.mono, fontSize: 10,
                }}>
                  <span style={{ color: C.text, fontWeight: 600, fontFamily: font.sans, fontSize: 11 }}>{b.player}</span>
                  <span style={{ color: C.textMuted }}>U{b.line} <span style={{ color: ec(b.edge), fontWeight: 700 }}>+{b.edge.toFixed(1)}</span></span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ flex: 1, overflow: "auto", padding: "10px 14px" }}>
          <div style={{ fontSize: 9, fontFamily: font.mono, color: C.textMuted, letterSpacing: "0.06em", marginBottom: 6 }}>
            2-LEG PAIRS <span style={{ color: C.textSoft }}>({pairs.length})</span>
          </div>
          {pairs.length === 0 ? (
            <div style={{ fontSize: 10, fontFamily: font.mono, color: C.textDim }}>2+ legs for pairs</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {pairs.map(([a, b], i) => {
                const same = a.team === b.team || a.team === b.opp || b.team === a.opp;
                return (
                  <div key={i} style={{
                    padding: "4px 8px", borderRadius: 5,
                    background: same ? C.red + "06" : C.green + "04",
                    borderLeft: `2px solid ${same ? C.red + "30" : C.green + "20"}`,
                    opacity: same ? 0.4 : 1,
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontFamily: font.mono, fontSize: 10 }}>
                      <span style={{ color: C.textSoft }}>{a.player} <span style={{ color: C.textDim }}>×</span> {b.player}</span>
                      <span style={{ color: C.green, fontWeight: 700 }}>+{(a.edge + b.edge).toFixed(1)}</span>
                    </div>
                    {same && <div style={{ fontFamily: font.mono, fontSize: 8, color: C.red, marginTop: 1 }}>Correlated</div>}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div style={{
          padding: "8px 14px", borderTop: `1px solid ${C.glassBorder}`,
          fontFamily: font.mono, fontSize: 9, color: C.textMuted,
          display: "flex", justifyContent: "space-between",
        }}>
          <span>{selected.length} singles</span>
          <span>{pairs.filter(([a, b]) => !(a.team === b.team || a.team === b.opp || b.team === a.opp)).length} valid pairs</span>
        </div>
      </div>
    </div>
  );

  // ─── PERFORMANCE TAB ───────────────────────────────────────────
  const Perf = () => (
    <div style={{ height: "calc(100vh - 52px)", overflow: "auto", padding: 12 }}>
      {/* Strategy cards with rings */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 10 }}>
        {Object.entries(performanceData).map(([key, s]) => (
          <div key={key} style={{
            ...glass({
              padding: "18px 16px 14px",
              opacity: s.status === "BENCHED" ? 0.5 : 1,
              boxShadow: s.status === "ACTIVE" ? `0 0 40px ${s.colour}06` : "none",
            }),
          }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
              <div>
                <div style={{ fontFamily: font.mono, fontSize: 10, color: s.colour, letterSpacing: "0.08em", fontWeight: 600, marginBottom: 2 }}>{key}</div>
                <div style={{ fontFamily: font.sans, fontSize: 12, color: C.textSoft }}>{s.name}</div>
              </div>
              <Pill color={s.status === "ACTIVE" ? C.green : C.red}>{s.status}</Pill>
            </div>

            {/* Ring + stats */}
            <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
              <Ring value={s.winRate} color={s.colour} size={96} label="WIN %" />
              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 14px" }}>
                  {[
                    { l: "BETS", v: s.totalBets, c: C.text },
                    { l: "ROI", v: `${s.roi > 0 ? "+" : ""}${s.roi}%`, c: s.roi > 0 ? C.green : C.red },
                    { l: "PUSH", v: `${s.pushRate}%`, c: C.amber },
                    { l: "STREAK", v: s.streak, c: s.streak[0] === "+" ? C.green : C.red },
                  ].map(({ l, v, c }) => (
                    <div key={l}>
                      <div style={{ fontFamily: font.mono, fontSize: 8, color: C.textDim, letterSpacing: "0.08em" }}>{l}</div>
                      <div style={{ fontFamily: font.num, fontSize: 14, color: c, fontWeight: 700, lineHeight: 1.2 }}>{v}</div>
                    </div>
                  ))}
                </div>
                <div style={{
                  fontFamily: font.mono, fontSize: 9, color: C.textMuted,
                  padding: "3px 6px", borderRadius: 4, background: s.colour + "08",
                }}>{s.threshold}</div>
              </div>
            </div>

            {/* Mini bar */}
            <div style={{ display: "flex", height: 3, marginTop: 12, gap: 1, borderRadius: 2, overflow: "hidden" }}>
              <div style={{ flex: s.winRate, background: C.green }} />
              <div style={{ flex: s.pushRate, background: C.amber }} />
              <div style={{ flex: s.lossRate, background: C.red }} />
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
        {/* Win % chart */}
        <div style={{ ...glass({ padding: "14px" }) }}>
          <div style={{ fontFamily: font.mono, fontSize: 9, color: C.textMuted, letterSpacing: "0.06em", marginBottom: 10 }}>
            CUMULATIVE WIN %
          </div>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
              <XAxis dataKey="round" type="number" domain={[1, 12]}
                tick={{ fontSize: 9, fontFamily: font.mono, fill: C.textDim }}
                axisLine={{ stroke: C.glassBorder }} tickLine={false} ticks={[1,3,5,7,9,11]} />
              <YAxis domain={[40, 100]}
                tick={{ fontSize: 9, fontFamily: font.mono, fill: C.textDim }}
                axisLine={false} tickLine={false} ticks={[50,65,80,95]} />
              <ReferenceLine y={75} stroke={C.textDim} strokeDasharray="2 6" />
              <Tooltip contentStyle={{ background: C.glass, border: `1px solid ${C.glassBorder}`, fontFamily: font.mono, fontSize: 10, borderRadius: 6, padding: "8px 10px" }} />
              {Object.entries(performanceData).map(([k, s]) => (
                <Line key={k} data={s.rounds} dataKey="cumWinPct"
                  stroke={s.colour} strokeWidth={s.status === "BENCHED" ? 1 : 2}
                  strokeDasharray={s.status === "BENCHED" ? "3 3" : "0"}
                  dot={{ r: 1.5, fill: s.colour }} name={k} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Profit chart */}
        <div style={{ ...glass({ padding: "14px" }) }}>
          <div style={{ fontFamily: font.mono, fontSize: 9, color: C.textMuted, letterSpacing: "0.06em", marginBottom: 10 }}>
            CUMULATIVE PROFIT (UNITS)
          </div>
          <ResponsiveContainer width="100%" height={170}>
            <AreaChart margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
              <defs>
                {Object.entries(performanceData).map(([k, s]) => (
                  <linearGradient key={k} id={`g-${k}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={s.colour} stopOpacity={0.15} />
                    <stop offset="100%" stopColor={s.colour} stopOpacity={0} />
                  </linearGradient>
                ))}
              </defs>
              <XAxis dataKey="round" type="number" domain={[1, 12]}
                tick={{ fontSize: 9, fontFamily: font.mono, fill: C.textDim }}
                axisLine={{ stroke: C.glassBorder }} tickLine={false} ticks={[1,3,5,7,9,11]} />
              <YAxis tick={{ fontSize: 9, fontFamily: font.mono, fill: C.textDim }}
                axisLine={false} tickLine={false} />
              <ReferenceLine y={0} stroke={C.red + "25"} strokeDasharray="2 4" />
              <Tooltip contentStyle={{ background: C.glass, border: `1px solid ${C.glassBorder}`, fontFamily: font.mono, fontSize: 10, borderRadius: 6 }} />
              {Object.entries(performanceData).map(([k, s]) => (
                <Area key={k} data={s.rounds} dataKey="profit" type="monotone"
                  stroke={s.colour} strokeWidth={s.status === "BENCHED" ? 1 : 2}
                  strokeDasharray={s.status === "BENCHED" ? "3 3" : "0"}
                  fill={`url(#g-${k})`} name={k} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Round table */}
      <div style={{ ...glass({ padding: "14px" }) }}>
        <div style={{ fontFamily: font.mono, fontSize: 9, color: C.textMuted, letterSpacing: "0.06em", marginBottom: 8 }}>
          ROUND-BY-ROUND
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: font.mono, fontSize: 10 }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${C.glassBorder}` }}>
              <th style={{ padding: "5px 6px", textAlign: "left", color: C.textMuted, fontSize: 9 }}>RND</th>
              {Object.entries(performanceData).map(([k, s]) => (
                <th key={k} style={{ padding: "5px 6px", textAlign: "center", color: s.colour, fontSize: 9, borderLeft: `1px solid ${C.glassBorder}50` }}>
                  {k} <span style={{ color: C.textDim, fontWeight: 400 }}>W/L/P</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {performanceData.S1.rounds.map((_, i) => (
              <tr key={i} style={{ borderBottom: `1px solid ${C.glassBorder}30` }}>
                <td style={{ padding: "4px 6px", color: C.textMuted, fontWeight: 600 }}>R{i + 1}</td>
                {Object.entries(performanceData).map(([k, s]) => {
                  const r = s.rounds[i];
                  return (
                    <td key={k} style={{ padding: "4px 6px", textAlign: "center", borderLeft: `1px solid ${C.glassBorder}30` }}>
                      <span style={{ color: C.green }}>{r.wins}</span>
                      <span style={{ color: C.textDim }}>/</span>
                      <span style={{ color: C.red }}>{r.losses}</span>
                      <span style={{ color: C.textDim }}>/</span>
                      <span style={{ color: C.amber }}>{r.pushes}</span>
                      <span style={{ color: C.textDim, marginLeft: 6, fontSize: 9 }}>{r.winPct.toFixed(0)}%</span>
                      <svg width={32} height={10} viewBox="0 0 32 10" style={{ marginLeft: 4, verticalAlign: "middle" }}>
                        {s.rounds.slice(0, i + 1).length > 1 && (() => {
                          const v = s.rounds.slice(0, i + 1).map(d => d.cumWinPct);
                          const mn = Math.min(...v, 40), mx = Math.max(...v, 100), rn = mx - mn || 1;
                          const pts = v.map((val, j) => `${(j / Math.max(v.length - 1, 1)) * 32},${10 - ((val - mn) / rn) * 8 - 1}`).join(" ");
                          return <polyline points={pts} fill="none" stroke={s.colour} strokeWidth="1" />;
                        })()}
                      </svg>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  // ─── SHELL ─────────────────────────────────────────────────────
  return (
    <div style={{ width: "100%", height: "100vh", background: C.bg, color: C.text, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Top bar */}
      <div style={{
        height: 52, display: "flex", alignItems: "center", padding: "0 16px",
        borderBottom: `1px solid ${C.glassBorder}`, background: C.surface, gap: 14, flexShrink: 0,
      }}>
        <span style={{ fontFamily: font.mono, fontSize: 13, fontWeight: 700, color: C.text, letterSpacing: "0.04em" }}>
          Tackle Unders
        </span>
        <span style={{ fontFamily: font.mono, fontSize: 10, color: C.textMuted }}>R{ROUND} · AFL 2025</span>

        <div style={{ display: "flex", gap: 2, marginLeft: 20 }}>
          {[{ k: "picks", l: "Picks" }, { k: "performance", l: "Performance" }].map(t => (
            <button key={t.k} onClick={() => setTab(t.k)} className="dg-btn" style={{
              padding: "5px 16px", fontFamily: font.mono, fontSize: 11, fontWeight: 500,
              cursor: "pointer", borderRadius: 6,
              background: tab === t.k ? "#ffffff0a" : "transparent",
              border: `1px solid ${tab === t.k ? "#ffffff12" : "transparent"}`,
              color: tab === t.k ? C.text : C.textMuted,
            }}>{t.l}</button>
          ))}
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 12, alignItems: "center" }}>
          {Object.entries(performanceData).map(([k, s]) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 5, fontFamily: font.mono, fontSize: 9, color: C.textMuted }}>
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: s.status === "ACTIVE" ? s.colour : C.red + "80",
                boxShadow: s.status === "ACTIVE" ? `0 0 6px ${s.colour}60` : "none",
              }} />
              {k}
            </div>
          ))}
        </div>
      </div>

      {tab === "picks" ? <Picks /> : <Perf />}
    </div>
  );
}
