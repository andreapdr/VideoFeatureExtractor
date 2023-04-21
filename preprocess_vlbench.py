import json
import csv
from os.path import expanduser
import os


def _process_times(times, if_none=0):
    processed = []
    for t in times:
        if t is None:
            processed.append(if_none)
        else:
            processed.append(int(t))
    return processed


def convert_multi3bench(input_file, output_file, output_dir, video_dirs):
    import pandas as pd

    with open(input_file, "r") as f:
        data = json.load(f)
    print(f"- converting {input_file}")

    video_paths = []
    feature_paths = []
    start_times = []
    end_times = []

    for k, v in data.items():
        dataset = v["dataset"]
        video_name = v.get("youtube_id", None)

        if video_name is None:
            if dataset == "star":
                video_name = v["video_file"] + ".mp4"
            else:
                video_name = v["video_file"] + ".webm"
        else:
            video_name = video_name + ".mp4"

        start_time = v["start_time"]
        end_time = v["end_time"]

        video_path = os.path.join(video_dirs[dataset], video_name)
        out_path = os.path.join(
            output_dir, "processed", video_name.split(".")[0] + ".npy"
        )

        video_paths.append(video_path)
        feature_paths.append(out_path)
        start_times.append(start_time)
        end_times.append(end_time)

    start_times = _process_times(start_times, if_none=0)
    end_times = _process_times(end_times, if_none=-1)
    data = {
        "video_path": video_paths,
        "feature_path": feature_paths,
        "start_time": start_times,
        "end_time": end_times,
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    return


def old_convert_multi3bench(input_file, output_file, output_dir, video_dirs):
    with open(input_file, "r") as f:
        data = json.load(f)
    csvfile = open(output_file, "w")
    writer = csv.writer(csvfile)
    video_paths = []
    feature_paths = []
    start_times = []
    end_times = []

    writer.writerow(["video_path", "feature_path", "start_time", "end_time"])

    for k, v in data.items():
        dataset = v["dataset"]
        if dataset == "ikea_asm":
            video_fname = v["video_file"] + ".mp4"
            video_path = os.path.join(video_dirs[dataset], video_fname)
        elif dataset == "something-something-v2":
            video_fname = v["video_file"] + ".webm"
            video_path = os.path.join(video_dirs[dataset], video_fname)
        elif dataset == "coin":
            video_fname = v["youtube_id"] + ".mp4"
            video_path = os.path.join(video_dirs[dataset], video_fname)
        elif dataset == "star":
            video_fname = v["video_file"] + ".mp4"
            video_path = os.path.join(video_dirs[dataset], video_fname)
        elif dataset == "rareact":
            video_fname = v["youtube_id"] + ".mp4"
            video_path = os.path.join(video_dirs[dataset], video_fname)

        out_path = os.path.join(
            output_dir, "processed", video_fname.split(".")[0] + ".npy"
        )

        start_time = v["start_time"]
        if start_time is None or v["dataset"] == "something-something-v2":
            start_time = 0
            end_time = -1
        else:
            start_time = int(v["start_time"])
            end_time = int(v["end_time"])

        video_paths.append(video_path)
        feature_paths.append(out_path)
        start_times.append(start_time)
        end_times.append(end_time)

        writer.writerow([video_path, out_path, start_time, end_time])
    csvfile.close()
    return


if __name__ == "__main__":
    input_file = "~/datasets/vl-bench/cos-balanced.filtered.json"
    video_dir = "~/datasets/vl-bench/videos/"
    output_file = "./_videos.csv"
    convert_multi3bench(expanduser(input_file), output_file, expanduser(video_dir))
