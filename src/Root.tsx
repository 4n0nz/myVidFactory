import {Composition} from "remotion";
import {Video} from "./Video";
import {SubsTerminal} from "./SubsTerminal";
import {MatrixRain} from "./MatrixRain";
import {BrowserSearch} from "./BrowserSearch";
import {ActionScene} from "./ActionScene";
import cues from "../cues.json";
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
      <Composition
        id="MatrixRain"
        component={MatrixRain}
        durationInFrames={300}
        fps={30}
        width={1280}
        height={720}
      />
      <Composition
        id="BrowserSearch"
        component={BrowserSearch}
        durationInFrames={360}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="ActionScene"
        component={ActionScene}
        durationInFrames={Math.ceil((((cues as any).audioDur ?? 12) + 0.6) * 30)}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
