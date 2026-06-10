import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, FONT} from "../theme";

export const Bullets: React.FC<{heading: string; bullets: string[]}> = ({
  heading,
  bullets,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const headO = interpolate(frame, [0, 16], [0, 1], {extrapolateRight: "clamp"});
  const headX = interpolate(frame, [0, 16], [-40, 0], {extrapolateRight: "clamp"});

  return (
    <AbsoluteFill style={{justifyContent: "center", padding: "0 160px"}}>
      <div style={{opacity: headO, transform: `translateX(${headX}px)`, marginBottom: 60, display: "flex", alignItems: "center", gap: 24}}>
        <div style={{width: 10, height: 70, background: COLORS.accent, borderRadius: 4}} />
        <div style={{fontFamily: FONT, fontSize: 64, fontWeight: 800, color: COLORS.text}}>{heading}</div>
      </div>
      <div style={{display: "flex", flexDirection: "column", gap: 30}}>
        {bullets.map((b, i) => {
          const start = 18 + i * 9;
          const s = spring({frame: frame - start, fps, config: {damping: 200}});
          const o = interpolate(s, [0, 1], [0, 1]);
          const x = interpolate(s, [0, 1], [60, 0]);
          return (
            <div
              key={i}
              style={{
                opacity: o,
                transform: `translateX(${x}px)`,
                display: "flex",
                alignItems: "center",
                gap: 26,
                fontFamily: FONT,
                fontSize: 40,
                color: COLORS.text,
              }}
            >
              <span style={{color: COLORS.accent2, fontSize: 32}}>▸</span>
              <span>{b}</span>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
