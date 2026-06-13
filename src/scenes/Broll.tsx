import {AbsoluteFill, OffthreadVideo, staticFile, interpolate, useCurrentFrame, useVideoConfig} from "remotion";
import {MONO} from "../theme";

const rnd = (a: number, b: number) => {
  const x = Math.sin(a * 91.3 + b * 47.7) * 43758.5453;
  return x - Math.floor(x);
};

// B-roll "coupure de signal" : clip plein écran muet + glitch à l'entrée/sortie.
export const Broll: React.FC<{clip: string}> = ({clip = "broll_00.mp4"}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const G = 10; // longueur du glitch (frames)

  // intensité glitch : fort aux extrémités, nul au milieu
  const gIn = interpolate(frame, [0, G], [1, 0], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  const gOut = interpolate(frame, [durationInFrames - G, durationInFrames], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  const g = Math.max(gIn, gOut);

  const jitterX = (rnd(frame, 1) - 0.5) * 26 * g;
  const jitterY = (rnd(frame, 2) - 0.5) * 10 * g;
  const flash = g > 0.6 && rnd(frame, 3) > 0.5 ? (g - 0.6) * 1.6 : 0;
  const tc = String(Math.floor(frame / 30)).padStart(2, "0");
  const blink = Math.floor(frame / 15) % 2 === 0;

  // barres de déchirure (datamosh)
  const bars = g > 0.05 ? Array.from({length: 6}).map((_, i) => {
    const y = rnd(frame, i + 10) * 100;
    const h = 2 + rnd(frame, i + 20) * 5;
    const dx = (rnd(frame, i + 30) - 0.5) * 80 * g;
    return {y, h, dx, c: rnd(frame, i + 40) > 0.5 ? "#00ff66" : "#0a1a0f"};
  }) : [];

  return (
    <AbsoluteFill style={{backgroundColor: "#000"}}>
      <AbsoluteFill style={{transform: `translate(${jitterX}px, ${jitterY}px) scale(${1 + g * 0.04})`}}>
        <OffthreadVideo src={staticFile(`broll/${clip}`)} muted
          style={{width: "100%", height: "100%", objectFit: "cover"}} />
      </AbsoluteFill>

      {/* teinte + scanlines + vignette */}
      <AbsoluteFill style={{background: "rgba(0,255,80,0.14)", mixBlendMode: "overlay"}} />
      <AbsoluteFill style={{backgroundImage: "repeating-linear-gradient(0deg, rgba(0,0,0,0.18) 0px, rgba(0,0,0,0.18) 1px, transparent 1px, transparent 3px)"}} />
      <AbsoluteFill style={{boxShadow: "inset 0 0 280px rgba(0,0,0,0.85)"}} />

      {/* barres glitch */}
      {bars.map((b, i) => (
        <div key={i} style={{position: "absolute", left: 0, right: 0, top: `${b.y}%`, height: b.h, background: b.c, transform: `translateX(${b.dx}px)`, opacity: 0.8 * g, mixBlendMode: "screen"}} />
      ))}
      {/* flash */}
      <AbsoluteFill style={{background: "#bfffce", opacity: flash}} />

      {/* habillage REC */}
      <AbsoluteFill style={{border: "2px solid rgba(0,255,0,0.35)", margin: 24}} />
      <div style={{position: "absolute", top: 44, left: 48, display: "flex", alignItems: "center", gap: 12, fontFamily: MONO, fontSize: 26, color: "#00ff00", textShadow: "0 0 8px rgba(0,255,0,0.6)", opacity: 1 - flash}}>
        <span style={{opacity: blink ? 1 : 0.2}}>◉</span> REC
      </div>
      <div style={{position: "absolute", top: 44, right: 48, fontFamily: MONO, fontSize: 26, color: "#00ff00", textShadow: "0 0 8px rgba(0,255,0,0.6)", opacity: 1 - flash}}>
        ░ SIGNAL · 00:{tc}
      </div>
    </AbsoluteFill>
  );
};
