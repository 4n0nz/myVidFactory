import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, FONT, MONO} from "../theme";

export const Title: React.FC<{title: string; subtitle: string; badge: string}> = ({
  title,
  subtitle,
  badge,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame, fps, config: {damping: 200}});
  const scale = interpolate(s, [0, 1], [0.85, 1]);
  const subO = interpolate(frame, [12, 30], [0, 1], {extrapolateRight: "clamp"});
  const badgeO = interpolate(frame, [24, 42], [0, 1], {extrapolateRight: "clamp"});
  const lineW = interpolate(s, [0, 1], [0, 520]);

  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center"}}>
      <div style={{transform: `scale(${scale})`, textAlign: "center"}}>
        <div
          style={{
            fontFamily: FONT,
            fontSize: 110,
            fontWeight: 800,
            letterSpacing: 8,
            color: COLORS.text,
            textShadow: "0 0 40px rgba(0,255,0,0.55)",
          }}
        >
          {title}
        </div>
        <div
          style={{
            height: 4,
            width: lineW,
            margin: "28px auto",
            background: "linear-gradient(90deg, transparent, #00ff00, transparent)",
          }}
        />
        <div
          style={{
            fontFamily: FONT,
            fontSize: 38,
            color: COLORS.dim,
            opacity: subO,
          }}
        >
          {subtitle}
        </div>
        <div
          style={{
            marginTop: 48,
            fontFamily: MONO,
            fontSize: 26,
            color: "#00ff00",
            opacity: badgeO,
            border: "1px solid #00ff00",
            borderRadius: 10,
            padding: "12px 26px",
            display: "inline-block",
            background: COLORS.panel,
          }}
        >
          {badge}
        </div>
      </div>
    </AbsoluteFill>
  );
};
