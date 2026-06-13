import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, FONT, MONO} from "./theme";

type Result = {title: string; url: string; snippet: string};

const DEFAULT_RESULTS: Result[] = [
  {title: "ShadowBroker — Real-Time Geospatial Intelligence Platform",
   url: "github.com › BigBodyCobain › Shadowbroker",
   snippet: "Open-source OSINT platform aggregating 60+ live intelligence feeds — aircraft, ships, satellites, CCTV — onto a single map interface."},
  {title: "ShadowBroker: an open-source OSINT map of everything",
   url: "news.ycombinator.com › item",
   snippet: "Track private jets, 25,000+ AIS vessels and 2,000+ satellites in real time. Built with Next.js, MapLibre GL and FastAPI."},
  {title: "How ShadowBroker turns public signals into one live map",
   url: "medium.com › osint › shadowbroker-deep-dive",
   snippet: "A look at the agentic command channel, the recon toolkit and the SAR ground-change detection layer."},
];

export const BrowserSearch: React.FC<{query?: string; results?: Result[]}> = ({
  query = "ShadowBroker OSINT platform",
  results = DEFAULT_RESULTS,
}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();

  // typing the query
  const typeStart = 12, typeEnd = 78;
  const typed = Math.floor(
    interpolate(frame, [typeStart, typeEnd], [0, query.length], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    })
  );
  const q = query.slice(0, typed);
  const caretOn = Math.floor(frame / 14) % 2 === 0;
  const typing = frame < typeEnd;

  // fake cursor: search bar -> first result
  const cx = interpolate(frame, [0, 70, 150, 230], [width * 0.62, width * 0.30, width * 0.30, width * 0.22], {extrapolateRight: "clamp", extrapolateLeft: "clamp"});
  const cy = interpolate(frame, [0, 70, 150, 230], [height * 0.55, height * 0.205, height * 0.205, height * 0.46], {extrapolateRight: "clamp", extrapolateLeft: "clamp"});

  return (
    <AbsoluteFill style={{backgroundColor: COLORS.bg, padding: 40}}>
      <div style={{
        width: "100%", height: "100%", borderRadius: 16, overflow: "hidden",
        border: `1px solid ${COLORS.border}`, background: "#0b0f14",
        display: "flex", flexDirection: "column",
        boxShadow: "0 30px 80px rgba(0,0,0,0.6)",
      }}>
        {/* chrome top bar */}
        <div style={{display: "flex", alignItems: "center", gap: 10, padding: "16px 22px", background: "#161b22", borderBottom: `1px solid ${COLORS.border}`}}>
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ff5f56"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ffbd2e"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#27c93f"}} />
          <div style={{marginLeft: 18, padding: "8px 18px", borderRadius: "10px 10px 0 0", background: "#0b0f14", color: COLORS.dim, fontFamily: FONT, fontSize: 22, maxWidth: 380, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis"}}>
            🔍 {query} — Recherche
          </div>
        </div>

        {/* toolbar + address bar */}
        <div style={{display: "flex", alignItems: "center", gap: 18, padding: "14px 24px", background: "#11161d", borderBottom: `1px solid ${COLORS.border}`}}>
          <span style={{color: COLORS.dim, fontSize: 26}}>←</span>
          <span style={{color: COLORS.dim, fontSize: 26}}>→</span>
          <span style={{color: COLORS.dim, fontSize: 24}}>⟳</span>
          <div style={{flex: 1, marginLeft: 8, padding: "12px 22px", borderRadius: 24, background: "#0b0f14", border: `1px solid ${COLORS.border}`, color: COLORS.text, fontFamily: MONO, fontSize: 22, display: "flex", alignItems: "center", gap: 12}}>
            <span style={{color: COLORS.accent}}>🔒</span>
            <span style={{color: COLORS.dim}}>https://www.google.com/search?q=</span>
            <span>{q.replace(/ /g, "+")}</span>
          </div>
        </div>

        {/* page */}
        <div style={{flex: 1, padding: "44px 90px", position: "relative"}}>
          {/* search field */}
          <div style={{display: "flex", alignItems: "center", gap: 18, width: 760, padding: "16px 26px", borderRadius: 30, background: "#11161d", border: `1px solid ${COLORS.accent}`, boxShadow: "0 0 18px rgba(0,255,0,0.18)"}}>
            <span style={{fontSize: 26}}>🔍</span>
            <span style={{fontFamily: FONT, fontSize: 30, color: COLORS.text}}>
              {q}{typing && caretOn ? <span style={{color: COLORS.accent}}>|</span> : null}
            </span>
          </div>

          {/* stats line */}
          <div style={{marginTop: 26, marginBottom: 30, fontFamily: FONT, fontSize: 19, color: COLORS.dim, opacity: interpolate(frame, [95, 110], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"})}}>
            Environ 9 240 résultats (0,38 secondes)
          </div>

          {/* results */}
          {results.map((r, i) => {
            const start = 112 + i * 20;
            const s = spring({frame: frame - start, fps, config: {damping: 200}});
            const o = interpolate(s, [0, 1], [0, 1]);
            const y = interpolate(s, [0, 1], [24, 0]);
            return (
              <div key={i} style={{opacity: o, transform: `translateY(${y}px)`, marginBottom: 34, maxWidth: 1100}}>
                <div style={{fontFamily: FONT, fontSize: 19, color: COLORS.accent2, marginBottom: 4}}>{r.url}</div>
                <div style={{fontFamily: FONT, fontSize: 30, color: "#8ab4f8", marginBottom: 6}}>{r.title}</div>
                <div style={{fontFamily: FONT, fontSize: 21, color: COLORS.dim, lineHeight: 1.5}}>{r.snippet}</div>
              </div>
            );
          })}

          {/* fake cursor */}
          <svg width="30" height="36" viewBox="0 0 30 36" style={{position: "absolute", left: cx, top: cy, filter: "drop-shadow(0 2px 3px rgba(0,0,0,0.6))"}}>
            <path d="M2 2 L2 28 L9 21 L14 32 L18 30 L13 19 L23 19 Z" fill="#fff" stroke="#000" strokeWidth="1.5" />
          </svg>
        </div>
      </div>
    </AbsoluteFill>
  );
};
