import {AbsoluteFill, Audio, Series, staticFile} from "remotion";
import manifest from "../render-manifest.json";
import {Background} from "./Background";
import {Title} from "./scenes/Title";
import {Bullets} from "./scenes/Bullets";
import {Architecture} from "./scenes/Architecture";
import {FileTree} from "./scenes/FileTree";
import {Code} from "./scenes/Code";
import {Terminal} from "./scenes/Terminal";
import {Chapter} from "./scenes/Chapter";
import {Stat} from "./scenes/Stat";

const MAP: Record<string, React.FC<any>> = {
  title: Title,
  bullets: Bullets,
  architecture: Architecture,
  filetree: FileTree,
  code: Code,
  terminal: Terminal,
  chapter: Chapter,
  stat: Stat,
};

export const Video: React.FC = () => {
  return (
    <AbsoluteFill>
      <Background />
      <Series>
        {manifest.scenes.map((scene: any) => {
          const Comp = MAP[scene.type];
          return (
            <Series.Sequence
              key={scene.id}
              durationInFrames={scene.durationInFrames}
            >
              <Comp {...scene.props} />
              {scene.audioFile ? (
                <Audio src={staticFile(scene.audioFile)} />
              ) : null}
            </Series.Sequence>
          );
        })}
      </Series>
    </AbsoluteFill>
  );
};
