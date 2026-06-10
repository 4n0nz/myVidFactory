import {AbsoluteFill, useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, MONO} from "./theme";
import captions from "../captions.json";

type Cap = {start: number; end: number; text: string};

// Live-transcript terminal: captions type in, older lines scroll up and dim.
// Transparent background so it composites as an alpha PiP overlay.
export const SubsTerminal: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;
  const caps = captions as Cap[];

  // index of the currently-speaking caption (last one whose start <= t)
  let active = -1;
  for (let i = 0; i < caps.length; i++) {
    if (caps[i].start <= t) active = i;
    else break;
  }

  // show the active line + enough previous ones to keep the 4-line window full
  const HISTORY = 8;
  const LINE = 26; // px per text line — fixed 4-line viewport
  const startIdx = Math.max(0, active - HISTORY);
  const visible = active >= 0 ? caps.slice(startIdx, active + 1) : [];

  const cursorOn = Math.floor(frame / 15) % 2 === 0;

  return (
    <AbsoluteFill style={{backgroundColor: "transparent"}}>
      <div
        style={{
          position: "absolute",
          left: 10,
          right: 10,
          bottom: 10,
          background: "rgba(5, 8, 13, 0.92)",
          border: "1px solid #00ff00",
          borderRadius: 10,
          boxShadow: "0 8px 28px rgba(0,0,0,0.6), 0 0 16px rgba(0,255,0,0.3)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* title bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "8px 12px",
            background: COLORS.panel2,
            borderBottom: `1px solid ${COLORS.border}`,
          }}
        >
          <div style={{width: 8, height: 8, borderRadius: "50%", background: "#ff5f56"}} />
          <div style={{width: 8, height: 8, borderRadius: "50%", background: "#ffbd2e"}} />
          <div style={{width: 8, height: 8, borderRadius: "50%", background: "#27c93f"}} />
          <div style={{marginLeft: 8, fontFamily: MONO, fontSize: 13, color: COLORS.dim}}>
            transcript ~ live
          </div>
        </div>

        {/* fixed 4-line viewport — text scrolls up, older lines clipped at top */}
        <div
          style={{
            height: LINE * 4,
            padding: "8px 14px",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-end",
          }}
        >
          {visible.map((c, i) => {
            const isActive = startIdx + i === active;
            let shown = c.text;
            if (isActive) {
              const prog = Math.min(1, Math.max(0, (t - c.start) / Math.max(0.3, c.end - c.start)));
              const n = Math.floor(prog * c.text.length);
              shown = c.text.slice(0, n);
            }
            return (
              <div
                key={startIdx + i}
                style={{
                  fontFamily: MONO,
                  fontSize: 16,
                  lineHeight: `${LINE}px`,
                  color: isActive ? COLORS.text : COLORS.dim,
                  opacity: isActive ? 1 : 0.5,
                }}
              >
                <span style={{color: COLORS.accent2}}>&gt; </span>
                {shown}
                {isActive && cursorOn && <span style={{color: COLORS.accent}}>▋</span>}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
