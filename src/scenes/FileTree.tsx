import {AbsoluteFill, interpolate, useCurrentFrame} from "remotion";
import {COLORS, FONT, MONO} from "../theme";

type Row = {name: string; depth: number; dir: boolean};

export const FileTree: React.FC<{root: string; tree: Row[]}> = ({root, tree}) => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center"}}>
      <div
        style={{
          width: 1100,
          background: COLORS.panel,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 14,
          padding: "40px 50px",
          boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        }}
      >
        <div style={{fontFamily: MONO, fontSize: 36, color: COLORS.accent, marginBottom: 24}}>
          📁 {root}
        </div>
        {tree.map((r, i) => {
          const start = 10 + i * 7;
          const o = interpolate(frame, [start, start + 8], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const x = interpolate(frame, [start, start + 8], [30, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={i}
              style={{
                opacity: o,
                transform: `translateX(${x}px)`,
                fontFamily: MONO,
                fontSize: 30,
                color: r.dir ? COLORS.text : COLORS.dim,
                paddingLeft: r.depth * 48 + 20,
                lineHeight: 1.95,
                fontWeight: r.dir ? 600 : 400,
              }}
            >
              <span style={{color: COLORS.border}}>{r.depth > 0 ? "└─ " : ""}</span>
              <span style={{color: r.dir ? COLORS.accent2 : COLORS.dim}}>{r.dir ? "▸ " : "  "}</span>
              {r.name}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
