import torchvision
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
        self,
        csv,
        framerate=1,
        size=112,
        centercrop=False,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def __len__(self):
        return len(self.csv)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_path = self.csv["video_path"].values[idx]
        output_file = self.csv["feature_path"].values[idx]
        start_time = self.csv["start_time"].values[idx]
        end_time = self.csv["end_time"].values[idx]

        if end_time != -1:
            if ".mp4" in video_path:
                segment_video_path = video_path.replace(
                    ".mp4", "_{}_{}.mp4".format(start_time, end_time)
                )
            elif ".avi" in video_path:
                segment_video_path = video_path.replace(
                    ".avi", "_{}_{}.avi".format(start_time, end_time)
                )
            else:
                raise ValueError(f"Video format not supported {video_path}")

            segment_output_file = output_file.replace(
                ".npy", "_{}_{}.npy".format(start_time, end_time)
            )

            if not os.path.exists(segment_output_file):
                # we have to rely on torchvision to cut the videos at the moment
                _video = torchvision.io.read_video(
                    video_path, pts_unit="sec", start_pts=start_time, end_pts=end_time
                )
                torchvision.io.write_video(
                    segment_video_path, _video[0], fps=_video[-1]["video_fps"]
                )

            video_path = segment_video_path
            output_file = segment_output_file

        video = th.zeros(1)

        if os.path.exists(output_file):
            # if output file exsits, we skip loading the video (retrub th.Tensor() to avoid error in collate_fn dataloader)
            return {"video": th.Tensor(), "input": video_path, "output": output_file}

        try:
            h, w = self._get_video_dim(video_path)

            height, width = self._get_output_dim(h, w)
            cmd = (
                ffmpeg.input(video_path)
                .filter("fps", fps=self.framerate)
                .filter("scale", width, height)
            )
            if self.centercrop:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
            out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                capture_stdout=True, quiet=True
            )
            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = th.from_numpy(video.astype("float32"))
            video = video.permute(0, 3, 1, 2)
        except:
            print("ffprobe failed at: {}".format(video_path))

        return {"video": video, "input": video_path, "output": output_file}
