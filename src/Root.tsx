import {Composition} from "remotion";
import {Video} from "./Video";
import {SubsTerminal} from "./SubsTerminal";
import manifest from "../render-manifest.json";
import captions from "../captions.json";

const total = manifest.scenes.reduce(
  (acc: number, s: any) => acc + s.durationInFrames,
  0
);

const fps = manifest.meta.fps;
const caps = captions as {start: number; end: number; text: string}[];
const subDur = caps.length
  ? Math.ceil((caps[caps.length - 1].end + 0.6) * fps)
  : 300;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="Tutorial"
        component={Video}
        durationInFrames={total}
        fps={fps}
        width={manifest.meta.width}
        height={manifest.meta.height}
      />
      <Composition
        id="SubsTerminal"
        component={SubsTerminal}
        durationInFrames={subDur}
        fps={fps}
        width={520}
        height={260}
      />
    </>
  );
};
