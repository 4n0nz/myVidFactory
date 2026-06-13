import {useCurrentFrame, useVideoConfig} from "remotion";
import {COLORS, MONO} from "../theme";

type Step = {cmd: string; output?: string[]};

const DEFAULT_STEPS: Step[] = [
  {cmd: "git clone https://github.com/BigBodyCobain/Shadowbroker.git",
   output: ["Cloning into 'Shadowbroker'...",
            "remote: Enumerating objects: 4821, done.",
            "Receiving objects: 100% (4821/4821), 41.2 MiB | 18.3 MiB/s, done.",
            "Resolving deltas: 100% (2680/2680), done."]},
  {cmd: "cd Shadowbroker", output: []},
  {cmd: "cp .env.example .env", output: ["# ajoute tes clés API (optionnel)"]},
  {cmd: "docker compose up -d",
   output: ["[+] Running 3/3",
            "  ✔ Network shadowbroker_default      Created",
            "  ✔ Container shadowbroker-backend     Started",
            "  ✔ Container shadowbroker-frontend    Started"]},
  {cmd: "# ✓ ShadowBroker en ligne → http://localhost:3000", output: []},
];

const CHARS_PER_SEC = 32;

export const Install: React.FC<{title?: string; steps?: Step[]}> = ({
  title = "anon@server: ~",
  steps = DEFAULT_STEPS,
}) => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const t = frame / fps;
  const total = durationInFrames / fps;

  // budget temps par étape : taper la cmd + révéler l'output + petite pause
  const cost = (s: Step) =>
    s.cmd.length / CHARS_PER_SEC + (s.output?.length ?? 0) * 0.35 + 0.5;
  const costs = steps.map(cost);
  const sum = costs.reduce((a, b) => a + b, 0);
  const speech = Math.max(total - 0.6, 1);
  const scale = speech / sum;

  // construire les lignes affichées jusqu'à `t`
  type Line = {kind: "cmd" | "out"; text: string; typing?: boolean};
  const lines: Line[] = [];
  let clock = 0;
  for (const s of steps) {
    const cmdDur = (s.cmd.length / CHARS_PER_SEC) * scale;
    const local = t - clock;
    if (local <= 0) break;
    const isComment = s.cmd.trimStart().startsWith("#");
    if (local < cmdDur) {
      const shown = Math.floor((local / cmdDur) * s.cmd.length);
      lines.push({kind: "cmd", text: s.cmd.slice(0, shown), typing: true});
      break;
    }
    lines.push({kind: "cmd", text: s.cmd, typing: false});
    // output révélé après la cmd
    const outLines = s.output ?? [];
    const outBudget = (outLines.length * 0.35 + 0.5) * scale;
    const afterCmd = local - cmdDur;
    const revealed = outBudget > 0 ? Math.min(outLines.length, Math.floor((afterCmd / outBudget) * outLines.length) + 1) : 0;
    for (let i = 0; i < (isComment ? 0 : Math.min(revealed, outLines.length)); i++) {
      lines.push({kind: "out", text: outLines[i]});
    }
    clock += cmdDur + outBudget;
  }

  const caretOn = Math.floor(frame / 14) % 2 === 0;
  const VISIBLE = 16; // lignes visibles (scroll)
  const shown = lines.slice(-VISIBLE);

  return (
    <div style={{position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center"}}>
      <div style={{width: 1500, height: 860, background: "#05080d", border: `1px solid ${COLORS.border}`, borderRadius: 16, overflow: "hidden", boxShadow: "0 30px 80px rgba(0,0,0,0.7)", display: "flex", flexDirection: "column"}}>
        <div style={{display: "flex", alignItems: "center", gap: 10, padding: "16px 24px", background: COLORS.panel2, borderBottom: `1px solid ${COLORS.border}`}}>
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ff5f56"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#ffbd2e"}} />
          <div style={{width: 14, height: 14, borderRadius: "50%", background: "#27c93f"}} />
          <div style={{marginLeft: 18, fontFamily: MONO, fontSize: 22, color: COLORS.dim}}>{title}</div>
        </div>
        <div style={{flex: 1, padding: "30px 40px", display: "flex", flexDirection: "column", justifyContent: "flex-end"}}>
          {shown.map((l, i) => {
            const last = i === shown.length - 1;
            const isComment = l.kind === "cmd" && l.text.trimStart().startsWith("#");
            return (
              <div key={i} style={{fontFamily: MONO, fontSize: 28, lineHeight: 1.7, whiteSpace: "pre-wrap", color: l.kind === "out" ? COLORS.dim : isComment ? COLORS.accent2 : COLORS.text}}>
                {l.kind === "cmd" && !isComment && <span style={{color: COLORS.accent}}>$ </span>}
                {l.text}
                {last && l.kind === "cmd" && l.typing && caretOn && <span style={{color: COLORS.accent}}>█</span>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
