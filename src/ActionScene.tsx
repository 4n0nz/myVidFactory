import {AbsoluteFill, Audio, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, FONT, MONO} from "./theme";
import cues from "../cues.json";
import {Background} from "./Background";

const RESULTS = [
  "geo · Frankfurt, DE  (50.11, 8.68)",
  "WHOIS · org SHADOW LLC",
  "OFAC sanctions · aucun match",
  "ports · 8000/tcp open (FastAPI)",
];

const Pointer: React.FC<{x: number; y: number}> = ({x, y}) => (
  <svg width="34" height="42" viewBox="0 0 30 36"
    style={{position: "absolute", left: x, top: y, filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.7))"}}>
    <path d="M2 2 L2 28 L9 21 L14 32 L18 30 L13 19 L23 19 Z" fill="#fff" stroke="#000" strokeWidth="1.5" />
  </svg>
);

export const ActionScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const sc = (cues as any).startClick * fps;   // clic Démarrer
  const ow = (cues as any).openWindow * fps;    // ouverture fenêtre
  const rc = (cues as any).resultClick * fps;   // clic résultat

  // cibles
  const BTN = {x: 690, y: 770};       // bouton Démarrer (centre approx)
  const RES = {x: 1340, y: 250};      // 1er résultat
  const IDLE = {x: 980, y: 560};

  // curseur : idle -> bouton (arrive à sc) -> reste -> résultat (arrive à rc)
  const cx = interpolate(frame, [0, sc - 16, sc, rc - 18, rc], [IDLE.x, BTN.x, BTN.x, BTN.x, RES.x], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  const cy = interpolate(frame, [0, sc - 16, sc, rc - 18, rc], [IDLE.y, BTN.y, BTN.y, BTN.y, RES.y], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});

  const pressed = (t: number) => frame >= t && frame < t + 7;
  const ripple = (t: number, x: number, y: number) => {
    if (frame < t || frame > t + 18) return null;
    const p = (frame - t) / 18;
    return <div style={{position: "absolute", left: x - 4, top: y - 4, width: 8 + p * 90, height: 8 + p * 90, marginLeft: -(p * 45), marginTop: -(p * 45), borderRadius: "50%", border: `2px solid ${COLORS.accent}`, opacity: 1 - p}} />;
  };

  const scanning = frame >= sc && frame < ow;
  const winS = spring({frame: frame - ow, fps, config: {damping: 200}});
  const resultHot = frame >= rc;

  return (
    <AbsoluteFill>
      <Background />
      {/* fenêtre principale — Recon Toolkit */}
      <div style={{position: "absolute", left: 80, top: 130, width: 820, height: 820, background: "#0b0f14", border: `1px solid ${COLORS.border}`, borderRadius: 14, overflow: "hidden", boxShadow: "0 30px 80px rgba(0,0,0,0.6)"}}>
        <div style={{display: "flex", alignItems: "center", gap: 10, padding: "16px 22px", background: "#161b22", borderBottom: `1px solid ${COLORS.border}`}}>
          <div style={{width: 13, height: 13, borderRadius: "50%", background: "#ff5f56"}} />
          <div style={{width: 13, height: 13, borderRadius: "50%", background: "#ffbd2e"}} />
          <div style={{width: 13, height: 13, borderRadius: "50%", background: "#27c93f"}} />
          <div style={{marginLeft: 16, fontFamily: FONT, fontSize: 22, color: COLORS.dim}}>ShadowBroker — Recon Toolkit</div>
        </div>
        <div style={{padding: "40px 44px"}}>
          <div style={{fontFamily: FONT, fontSize: 26, color: COLORS.text, marginBottom: 18}}>Cible</div>
          <div style={{padding: "16px 20px", borderRadius: 10, background: "#11161d", border: `1px solid ${COLORS.border}`, fontFamily: MONO, fontSize: 24, color: COLORS.accent2}}>152.89.44.10</div>
          <div style={{marginTop: 28, fontFamily: MONO, fontSize: 20, color: COLORS.dim, height: 120}}>
            {scanning ? "▶ analyse en cours" + ".".repeat(1 + (Math.floor(frame / 8) % 3)) : frame >= ow ? "✓ analyse terminée — 4 résultats" : "prêt."}
          </div>
        </div>
        {/* bouton Démarrer (bas-droite) */}
        <div style={{position: "absolute", left: 540, top: 720, width: 230, height: 76,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: pressed(sc) ? "#00cc00" : "#00ff00", color: "#06210a",
          borderRadius: 12, fontFamily: FONT, fontSize: 26, fontWeight: 800,
          transform: pressed(sc) ? "scale(0.95)" : "scale(1)",
          boxShadow: "0 0 22px rgba(0,255,0,0.4)"}}>
          ▶ Analyser
        </div>
      </div>

      {/* fenêtre résultats — s'ouvre au mot "fenêtre" */}
      {frame >= ow && (
        <div style={{position: "absolute", left: 960, top: 130, width: 880, height: 820, background: "#0b0f14", border: `1px solid ${COLORS.accent}`, borderRadius: 14, overflow: "hidden", boxShadow: "0 0 30px rgba(0,255,0,0.2)", opacity: winS, transform: `scale(${interpolate(winS, [0, 1], [0.92, 1])})`, transformOrigin: "center"}}>
          <div style={{display: "flex", alignItems: "center", gap: 10, padding: "16px 22px", background: "#161b22", borderBottom: `1px solid ${COLORS.border}`}}>
            <div style={{width: 13, height: 13, borderRadius: "50%", background: "#ff5f56"}} />
            <div style={{width: 13, height: 13, borderRadius: "50%", background: "#ffbd2e"}} />
            <div style={{width: 13, height: 13, borderRadius: "50%", background: "#27c93f"}} />
            <div style={{marginLeft: 16, fontFamily: FONT, fontSize: 22, color: COLORS.dim}}>Résultats OSINT</div>
          </div>
          <div style={{padding: "30px 36px"}}>
            {RESULTS.map((r, i) => {
              const hot = i === 0 && resultHot;
              return (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 16,
                  padding: "20px 22px", marginBottom: 16, borderRadius: 10,
                  background: hot ? "rgba(0,255,0,0.10)" : "#11161d",
                  border: `1px solid ${hot ? COLORS.accent : COLORS.border}`,
                  fontFamily: MONO, fontSize: 23, color: COLORS.text,
                }}>
                  <span style={{color: COLORS.accent2}}>{String(i + 1).padStart(2, "0")}</span>
                  {r}
                </div>
              );
            })}
            {resultHot && (
              <div style={{marginTop: 10, padding: "20px 22px", borderRadius: 10, background: "#11161d", border: `1px solid ${COLORS.border}`, fontFamily: MONO, fontSize: 20, color: COLORS.dim, opacity: interpolate(frame, [rc, rc + 12], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"})}}>
                ↳ dossier : ASN AS200000 · abuse@shadow.llc · first seen 2019
              </div>
            )}
          </div>
        </div>
      )}

      {ripple(sc, BTN.x, BTN.y)}
      {ripple(rc, RES.x, RES.y)}
      <Pointer x={cx} y={cy} />

      <Audio src={staticFile("demo_action.mp3")} />
    </AbsoluteFill>
  );
};
