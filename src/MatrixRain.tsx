import {AbsoluteFill, useCurrentFrame} from "remotion";

// Pluie de code vert sur fond NOIR — overlay pour blend "screen" (projection).
const CHARS = "アカサタナハマヤラ0123456789ABCDEF<>=/$#*+".split("");
const CW = 22;   // largeur colonne / taille char
const TAIL = 20; // longueur de la traîne

const rand = (a: number, b: number, c: number) => {
  const x = Math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453;
  return x - Math.floor(x);
};

export const MatrixRain: React.FC = () => {
  const frame = useCurrentFrame();
  const W = 1280, H = 720;
  const cols = Math.ceil(W / CW);
  const rows = Math.ceil(H / CW);
  const cells: React.ReactNode[] = [];

  for (let c = 0; c < cols; c++) {
    const speed = 0.25 + rand(c, 7, 0) * 0.55;       // vitesse colonne
    const offset = rand(c, 99, 0) * (rows + TAIL);
    const head = (frame * speed + offset) % (rows + TAIL);
    for (let t = 0; t < TAIL; t++) {
      const r = Math.floor(head) - t;
      if (r < 0 || r >= rows) continue;
      const lum = 1 - t / TAIL;                       // tête = vif, queue = sombre
      const ch = CHARS[Math.floor(rand(c, r, Math.floor(frame / 3)) * CHARS.length)];
      const color = t === 0 ? "#d6ffd6" : `rgba(0,255,70,${(lum * lum).toFixed(3)})`;
      cells.push(
        <span
          key={`${c}-${t}`}
          style={{
            position: "absolute",
            left: c * CW,
            top: r * CW,
            width: CW,
            height: CW,
            fontFamily: "monospace",
            fontSize: CW,
            lineHeight: `${CW}px`,
            textAlign: "center",
            color,
            textShadow: t === 0 ? "0 0 8px #00ff46" : "none",
          }}
        >
          {ch}
        </span>
      );
    }
  }

  return <AbsoluteFill style={{backgroundColor: "#000"}}>{cells}</AbsoluteFill>;
};
