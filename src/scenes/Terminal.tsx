import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, MONO} from "../theme";

export const Terminal: React.FC<{commands: string[]}> = ({commands}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // typing budget: distribute commands across the scene
  const charsPerSec = 26;
  let cursor = 0;
  const timed = commands.map((cmd) => {
    const startChar = cursor;
    cursor += cmd.length + 6; // gap between lines
    return {cmd, startFrame: (startChar / charsPerSec) * fps};
  });

  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center"}}>
      <div
        style={{
          width: 1300,
          minHeight: 520,
          background: "#05080d",
          border: `1px solid ${COLORS.border}`,
          borderRadius: 14,
          overflow: "hidden",
          boxShadow: "0 24px 70px rgba(0,0,0,0.6)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "18px 24px",
            background: COLORS.panel2,
          }}
        >
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ff5f56"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ffbd2e"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#27c93f"}} />
          <div style={{marginLeft: 18, fontFamily: MONO, fontSize: 22, color: COLORS.dim}}>
            anon@shadowbroker: ~
          </div>
        </div>
        <div style={{padding: "36px 40px"}}>
          {timed.map(({cmd, startFrame}, i) => {
            const local = frame - startFrame;
            const shownChars = Math.max(0, Math.floor(local / fps * charsPerSec));
            if (local < 0) return null;
            const text = cmd.slice(0, shownChars);
            const typing = shownChars < cmd.length;
            const isComment = cmd.trimStart().startsWith("#");
            const cursorOn = Math.floor(frame / 15) % 2 === 0;
            return (
              <div
                key={i}
                style={{
                  fontFamily: MONO,
                  fontSize: 30,
                  lineHeight: 2.0,
                  color: isComment ? COLORS.comment : COLORS.text,
                  whiteSpace: "pre-wrap",
                }}
              >
                {!isComment && <span style={{color: COLORS.accent2}}>$ </span>}
                {text}
                {typing && cursorOn && <span style={{color: COLORS.accent}}>▋</span>}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
