import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, FONT} from "../theme";

type Item = {value: string; label: string};

// extract leading number for count-up, keep suffix (e.g. "25 000+", "60+")
const parse = (v: string) => {
  const m = v.match(/^([\d\s.,]+)(.*)$/);
  if (!m) return {num: null as number | null, suffix: v};
  const num = parseFloat(m[1].replace(/[\s,]/g, ""));
  return {num: isNaN(num) ? null : num, suffix: m[2]};
};

const fmt = (n: number) =>
  Math.round(n).toLocaleString("fr-FR").replace(/ /g, " ");

export const Stat: React.FC<{items: Item[]}> = ({items}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center"}}>
      <div style={{display: "flex", gap: 90, justifyContent: "center", flexWrap: "wrap"}}>
        {items.map((it, i) => {
          const start = i * 10;
          const s = spring({frame: frame - start, fps, config: {damping: 200}});
          const o = interpolate(s, [0, 1], [0, 1]);
          const scale = interpolate(s, [0, 1], [0.6, 1]);
          const {num, suffix} = parse(it.value);
          const prog = interpolate(frame - start, [4, 30], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const display = num !== null ? fmt(num * prog) + suffix : it.value;
          return (
            <div key={i} style={{textAlign: "center", opacity: o, transform: `scale(${scale})`}}>
              <div
                style={{
                  fontFamily: FONT,
                  fontSize: 130,
                  fontWeight: 900,
                  color: COLORS.accent,
                  textShadow: `0 0 40px rgba(0,255,0,0.45)`,
                  lineHeight: 1,
                }}
              >
                {display}
              </div>
              <div style={{fontFamily: FONT, fontSize: 34, color: COLORS.dim, marginTop: 20, maxWidth: 340}}>
                {it.label}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
