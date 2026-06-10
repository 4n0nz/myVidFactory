import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, FONT, MONO} from "../theme";

type Node = {label: string; x: number};

export const Architecture: React.FC<{nodes: Node[]; side: string}> = ({nodes, side}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const sideO = interpolate(frame, [40, 58], [0, 1], {extrapolateRight: "clamp"});

  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center"}}>
      <div style={{display: "flex", alignItems: "center", gap: 0}}>
        {nodes.map((n, i) => {
          const start = i * 12;
          const s = spring({frame: frame - start, fps, config: {damping: 200}});
          const o = interpolate(s, [0, 1], [0, 1]);
          const y = interpolate(s, [0, 1], [40, 0]);
          const arrowStart = start + 8;
          const aw = interpolate(frame - arrowStart, [0, 10], [0, 80], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div key={i} style={{display: "flex", alignItems: "center"}}>
              {i > 0 && (
                <div style={{width: 80, display: "flex", alignItems: "center"}}>
                  <div style={{height: 3, width: aw, background: COLORS.accent}} />
                  <div style={{
                    width: 0, height: 0,
                    borderTop: "8px solid transparent",
                    borderBottom: "8px solid transparent",
                    borderLeft: `12px solid ${COLORS.accent}`,
                    opacity: aw > 70 ? 1 : 0,
                  }} />
                </div>
              )}
              <div
                style={{
                  opacity: o,
                  transform: `translateY(${y}px)`,
                  width: 320,
                  height: 200,
                  background: COLORS.panel,
                  border: `2px solid ${COLORS.accent}`,
                  borderRadius: 16,
                  boxShadow: "0 0 30px rgba(0,255,0,0.18)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  padding: 24,
                  fontFamily: FONT,
                  fontSize: 30,
                  fontWeight: 700,
                  color: COLORS.text,
                  whiteSpace: "pre-line",
                  lineHeight: 1.4,
                }}
              >
                {n.label}
              </div>
            </div>
          );
        })}
      </div>
      <div
        style={{
          marginTop: 70,
          opacity: sideO,
          fontFamily: MONO,
          fontSize: 26,
          color: COLORS.dim,
          border: `1px dashed ${COLORS.border}`,
          borderRadius: 10,
          padding: "16px 30px",
        }}
      >
        {side}
      </div>
    </AbsoluteFill>
  );
};
