import React from 'react';
import {Composition} from 'remotion';
import {MusicVideo, type MusicVideoProps} from './MusicVideo';

const FPS = 30;
const WIDTH = 1920;
const HEIGHT = 1080;

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="MusicVideo"
      component={MusicVideo}
      durationInFrames={FPS * 8}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      defaultProps={{
        fps: FPS,
        width: WIDTH,
        height: HEIGHT,
        durationInFrames: FPS * 8,
        audio: '',
        windows: [],
      }}
      calculateMetadata={async ({props}: {props: MusicVideoProps}) => {
        const durationInFrames = Math.max(
          2,
          Number(props.durationInFrames) ||
            (props.windows?.length
              ? Number(props.windows[props.windows.length - 1].end_frame)
              : FPS * 8),
        );
        return {
          durationInFrames,
          fps: Number(props.fps) || FPS,
          width: Number(props.width) || WIDTH,
          height: Number(props.height) || HEIGHT,
        };
      }}
    />
  );
};
