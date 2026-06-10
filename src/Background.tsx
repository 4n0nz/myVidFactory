import {AbsoluteFill, useCurrentFrame} from "remotion";
import {COLORS} from "./theme";

export const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const drift = (frame * 0.15) % 60;
  return (
    <AbsoluteFill style={{backgroundColor: COLORS.bg}}>
      <AbsoluteFill
        style={{
          backgroundImage: `linear-gradient(rgba(0,255,0,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,0,0.5) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
          backgroundPosition: `${drift}px ${drift}px`,
          opacity: 0.22,
        }}
      />
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 40%, rgba(0,255,0,0.10), transparent 60%)`,
        }}
      />
      <AbsoluteFill
        style={{
          boxShadow: "inset 0 0 300px rgba(0,0,0,0.9)",
        }}
      />
    </AbsoluteFill>
  );
};
