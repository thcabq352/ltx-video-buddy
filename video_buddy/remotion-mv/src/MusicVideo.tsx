import React from 'react';
import {AbsoluteFill, Audio, OffthreadVideo, Sequence} from 'remotion';

export type MusicVideoWindow = {
  index: number;
  start_frame: number;
  end_frame: number;
  clip: string;
};

export type MusicVideoProps = {
  fps: number;
  width: number;
  height: number;
  durationInFrames: number;
  audio: string;
  windows: MusicVideoWindow[];
};

/**
 * MTV-style cut: one Sequence per beat window, full-track audio underneath.
 * Clips are unique Comfy/LTX motion burns — never still-holds.
 */
export const MusicVideo: React.FC<MusicVideoProps> = ({windows, audio}) => {
  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      {(windows || []).map((window) => {
        const durationInFrames = Math.max(
          1,
          Number(window.end_frame) - Number(window.start_frame),
        );
        return (
          <Sequence
            key={window.index}
            from={Number(window.start_frame)}
            durationInFrames={durationInFrames}
          >
            <AbsoluteFill>
              <OffthreadVideo
                src={window.clip}
                style={{width: '100%', height: '100%', objectFit: 'cover'}}
              />
            </AbsoluteFill>
          </Sequence>
        );
      })}
      {audio ? <Audio src={audio} /> : null}
    </AbsoluteFill>
  );
};
