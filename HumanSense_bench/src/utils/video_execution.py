import os
import subprocess


def split_video(video_file, start_time, end_time, tmp_video_dir):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(tmp_video_dir, "tmp_60")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    current_working_directory = os.getcwd()
    output_file = os.path.join(current_working_directory, output_file)

    # print("*"*30)
    # print(absolute_output_dir)

    if os.path.exists(output_file):
        print(f"Video file {output_file} already exists.")
        return output_file

    # or /path/to/ffmpeg
    FFMPEG_PATH = "ffmpeg"

    duration = float(end_time) - float(start_time)
    cmd = [
    FFMPEG_PATH,
    "-ss", str(start_time),
    "-i", video_file,
    "-t", str(duration),
    "-vcodec", "libx264",
    "-acodec", "aac",
    "-y",  # overwrite
    output_file
        ]

    # print(cmd)   
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ FFmpeg failed:")
        print(result.stderr)
        raise RuntimeError("FFmpeg cut failed")
    return output_file
    