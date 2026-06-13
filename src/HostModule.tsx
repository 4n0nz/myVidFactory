import {AbsoluteFill, OffthreadVideo, staticFile, interpolate, useCurrentFrame} from "remotion";
import manifest from "../render-manifest.json";
import captions from "../captions.json";
import {COLORS, MONO} from "./theme";

type Cap = {start: number; end: number; text: string};
const caps = captions as Cap[];
const fps = manifest.meta.fps;

// segments de scènes avec leur mode host
type Seg = {start: number; end: number; mode: "hero" | "pip" | "off"};
const SEGS: Seg[] = [];
{
  let acc = 0;
  for (const s of manifest.scenes as any[]) {
    const mode = s.host || (s.props && s.props.host) || "pip";
    SEGS.push({start: acc, end: acc + s.durationInFrames, mode});
    acc += s.durationInFrames;
  }
}

// dimensions internes du module (référence, échelle k=1 = HERO)
const UW = 820;                       // largeur unité
const AVH = Math.round(UW * 464 / 848); // hauteur avatar (ratio source)
const GAP = 14;

// layouts : translate (coin haut-gauche) + échelle + opacité
const HK = 1.42;                          // échelle HERO (plus gros, remplit l'écran)
const HERO = {tx: (1920 - UW * HK) / 2, ty: 72, k: HK, op: 1};
const PIP = {tx: 1920 - UW * 0.6 - 48, ty: 1080 - (AVH + GAP + 250) * 0.6 - 48, k: 0.6, op: 1};
const OFF = {tx: PIP.tx, ty: PIP.ty, k: 0.6, op: 0};
const lay = (m: string) => (m === "hero" ? HERO : m === "off" ? OFF : PIP);
const mix = (a: any, b: any, p: number) => ({
  tx: a.tx + (b.tx - a.tx) * p, ty: a.ty + (b.ty - a.ty) * p,
  k: a.k + (b.k - a.k) * p, op: a.op + (b.op - a.op) * p,
});
const ease = (p: number) => p * p * (3 - 2 * p);

export const HostModule: React.FC = () => {
  const frame = useCurrentFrame();
  const R = 9; // demi-fenêtre de transition (frames)

  // état courant + transition aux frontières
  let i = SEGS.findIndex((s) => frame < s.end);
  if (i < 0) i = SEGS.length - 1;
  let L = lay(SEGS[i].mode);
  for (let j = 1; j < SEGS.length; j++) {
    const b = SEGS[j].start;
    if (SEGS[j].mode !== SEGS[j - 1].mode && frame >= b - R && frame <= b + R) {
      const p = ease((frame - (b - R)) / (2 * R));
      L = mix(lay(SEGS[j - 1].mode), lay(SEGS[j].mode), p);
      break;
    }
  }

  // terminal : captions défilantes (fenêtre 3 lignes)
  const t = frame / fps;
  let active = -1;
  for (let n = 0; n < caps.length; n++) { if (caps[n].start <= t) active = n; else break; }
  const HIST = 6, LINE = 30;
  const startIdx = Math.max(0, active - HIST);
  const visible = active >= 0 ? caps.slice(startIdx, active + 1) : [];
  const caretOn = Math.floor(frame / 14) % 2 === 0;

  return (
    <AbsoluteFill style={{pointerEvents: "none"}}>
      <div style={{position: "absolute", left: 0, top: 0, transformOrigin: "top left",
        transform: `translate(${L.tx}px, ${L.ty}px) scale(${L.k})`, opacity: L.op, width: UW}}>
        {/* avatar */}
        <div style={{width: UW, height: AVH, borderRadius: 12, overflow: "hidden",
          border: "2px solid #00ff00", boxShadow: "0 0 22px rgba(0,255,0,0.3)", background: "#000"}}>
          <OffthreadVideo src={staticFile("avatar.mp4")} muted loop
            style={{width: "100%", height: "100%", objectFit: "cover"}} />
        </div>
        {/* terminal collé dessous */}
        <div style={{marginTop: GAP, borderRadius: 10, overflow: "hidden",
          border: "1px solid #00ff00", background: "rgba(5,8,13,0.92)",
          boxShadow: "0 8px 28px rgba(0,0,0,0.6), 0 0 16px rgba(0,255,0,0.25)"}}>
          <div style={{display: "flex", alignItems: "center", gap: 6, padding: "8px 14px",
            background: COLORS.panel2, borderBottom: `1px solid ${COLORS.border}`}}>
            <div style={{width: 9, height: 9, borderRadius: "50%", background: "#ff5f56"}} />
            <div style={{width: 9, height: 9, borderRadius: "50%", background: "#ffbd2e"}} />
            <div style={{width: 9, height: 9, borderRadius: "50%", background: "#27c93f"}} />
            <div style={{marginLeft: 8, fontFamily: MONO, fontSize: 15, color: COLORS.dim}}>transcript ~ live</div>
          </div>
          <div style={{height: LINE * 3, padding: "8px 16px", overflow: "hidden",
            display: "flex", flexDirection: "column", justifyContent: "flex-end"}}>
            {visible.map((c, n) => {
              const isActive = startIdx + n === active;
              let shown = c.text;
              if (isActive) {
                const prog = Math.min(1, Math.max(0, (t - c.start) / Math.max(0.3, c.end - c.start)));
                shown = c.text.slice(0, Math.floor(prog * c.text.length));
              }
              return (
                <div key={startIdx + n} style={{fontFamily: MONO, fontSize: 19, lineHeight: `${LINE}px`,
                  color: isActive ? COLORS.text : COLORS.dim, opacity: isActive ? 1 : 0.5}}>
                  <span style={{color: COLORS.accent2}}>&gt; </span>{shown}
                  {isActive && caretOn && <span style={{color: COLORS.accent}}>▋</span>}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
