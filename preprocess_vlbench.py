import json
import csv
from os.path import expanduser
import os


def convert_multi3bench(input_file, output_file, video_dir="./videos"):
    with open(input_file, "r") as f:
        data = json.load(f)
    csvfile = open(output_file, "w")
    writer = csv.writer(csvfile)
    writer.writerow(["video_path", "feature_path"])

    for k, v in data.items():
        if v["youtube_id"] is not None:
            video_fname = v["youtube_id"] + ".mp4"
        else:
            video_fname = v["video_file"]
            if v["dataset"] == "smsm":
                video_fname += ".webm"
            elif v["dataset"] == "ikea":
                video_fname += ".avi"
            else:
                video_fname += ".mp4"

        video_path = os.path.join(video_dir, video_fname)
        out_path = f"cache/processed/{video_fname.split('.')[0]}.npy"
        writer.writerow([video_path, out_path])
    csvfile.close()
    return


if __name__ == "__main__":
    input_file = "~/datasets/vl-bench/cos-balanced.filtered.json"
    video_dir = "~/datasets/vl-bench/videos/"
    output_file = "./_videos.csv"
    convert_multi3bench(expanduser(input_file), output_file, expanduser(video_dir))
