import {AbsoluteFill, interpolate, useCurrentFrame} from "remotion";
import {COLORS, FONT, MONO} from "../theme";

const KEYWORDS = new Set([
  "from", "import", "class", "def", "return", "if", "else", "for", "in",
  "True", "False", "None", "async", "await", "name", "description",
  "version", "author", "true", "false",
]);

// very lightweight tokenizer — comments, strings, keywords
const renderLine = (line: string, lang: string) => {
  const commentChar = lang === "python" || lang === "yaml" ? "#" : "//";
  const ci = line.indexOf(commentChar);
  let code = line;
  let comment = "";
  if (ci >= 0) {
    code = line.slice(0, ci);
    comment = line.slice(ci);
  }
  // split keeping quotes
  const parts = code.split(/("[^"]*")/g);
  return (
    <>
      {parts.map((p, i) => {
        if (p.startsWith('"') && p.endsWith('"')) {
          return <span key={i} style={{color: COLORS.string}}>{p}</span>;
        }
        const words = p.split(/(\b)/);
        return (
          <span key={i}>
            {words.map((w, j) =>
              KEYWORDS.has(w) ? (
                <span key={j} style={{color: COLORS.keyword}}>{w}</span>
              ) : (
                <span key={j}>{w}</span>
              )
            )}
          </span>
        );
      })}
      {comment && <span style={{color: COLORS.comment}}>{comment}</span>}
    </>
  );
};

export const Code: React.FC<{
  filename: string;
  lang: string;
  code: string;
  highlight?: number[];
}> = ({filename, lang, code, highlight = []}) => {
  const frame = useCurrentFrame();
  const lines = code.split("\n");
  const hl = new Set(highlight);

  return (
    <AbsoluteFill style={{justifyContent: "center", alignItems: "center"}}>
      <div
        style={{
          width: 1300,
          background: COLORS.panel,
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
            borderBottom: `1px solid ${COLORS.border}`,
          }}
        >
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ff5f56"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ffbd2e"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#27c93f"}} />
          <div style={{marginLeft: 18, fontFamily: MONO, fontSize: 24, color: COLORS.dim}}>
            {filename}
          </div>
        </div>
        <div style={{padding: "30px 0"}}>
          {lines.map((ln, i) => {
            const start = 6 + i * 3.5;
            const o = interpolate(frame, [start, start + 6], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const highlighted = hl.has(i + 1);
            return (
              <div
                key={i}
                style={{
                  opacity: o,
                  display: "flex",
                  fontFamily: MONO,
                  fontSize: 27,
                  lineHeight: 1.7,
                  background: highlighted ? "rgba(0,255,0,0.10)" : "transparent",
                  borderLeft: highlighted
                    ? `4px solid ${COLORS.accent}`
                    : "4px solid transparent",
                  padding: "0 30px",
                }}
              >
                <span style={{color: COLORS.border, width: 50, display: "inline-block", textAlign: "right", marginRight: 24, userSelect: "none"}}>
                  {i + 1}
                </span>
                <span style={{color: COLORS.text, whiteSpace: "pre"}}>{renderLine(ln, lang)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
