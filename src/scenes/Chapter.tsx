import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, FONT} from "../theme";

export const Chapter: React.FC<{num: string; title: string}> = ({num, title}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame, fps, config: {damping: 200}});
  const numO = interpolate(s, [0, 1], [0, 1]);
  const numY = interpolate(s, [0, 1], [60, 0]);
  const titleO = interpolate(frame, [14, 30], [0, 1], {extrapolateRight: "clamp"});
  const lineW = interpolate(frame, [10, 36], [0, 700], {extrapolateRight: "clamp", extrapolateLeft: "clamp"});

  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center"}}>
      <div style={{textAlign: "center"}}>
        <div
          style={{
            fontFamily: FONT,
            fontSize: 220,
            fontWeight: 900,
            color: "transparent",
            WebkitTextStroke: `3px ${COLORS.accent}`,
            opacity: numO,
            transform: `translateY(${numY}px)`,
            lineHeight: 1,
            letterSpacing: 4,
          }}
        >
          {num}
        </div>
        <div
          style={{
            height: 4,
            width: lineW,
            margin: "30px auto",
            background: `linear-gradient(90deg, transparent, ${COLORS.accent2}, transparent)`,
          }}
        />
        <div
          style={{
            fontFamily: FONT,
            fontSize: 70,
            fontWeight: 800,
            color: COLORS.text,
            opacity: titleO,
            letterSpacing: 1,
          }}
        >
          {title}
        </div>
      </div>
    </AbsoluteFill>
  );
};
