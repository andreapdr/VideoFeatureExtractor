import json
import csv
from os.path import expanduser


def convert_multi3bench(input_file, output_file, video_dir="./videos"):
    with open(input_file, 'r') as f:
        data = json.load(f)
    csvfile = open(output_file, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(["video_path", "feature_path"])

    for k, v in data.items():
        video_path = f"{video_dir}/{v['original_dataset']}/{k}.mp4"
        out_path = f"processed/{k}.npy"
        writer.writerow([video_path, out_path])
    csvfile.close()
    return


if __name__ == "__main__":
    input_file = "~/presuppositions/changeOfState.json"
    video_dir = "~/ytdownload/downloads/processed"
    output_file = "./multi3bench_data.csv"
    convert_multi3bench(expanduser(input_file),
                        output_file, expanduser(video_dir))
