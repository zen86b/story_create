import argparse
from utils.video_create import create_video


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str,required=True, help="Path to input directory which contain image, voice and transcript")
parser.add_argument("--video-dim-h", type=int, default=1080, help="Height of output video")
parser.add_argument("--video-dim-w", type=int, default=1920, help="Width of output video")
parser.add_argument("--fps", type=int, default=60, help="Fps of video")
parser.add_argument("--speed", type=float, default=1.0, help="Speed of video")

# subtitile param
parser.add_argument("--add-sub", action="store_true", default=False, help="Enable adding subtitle to video")
parser.add_argument("--font-type", type=str, default="", help="Path to the font using to write subtitle")
parser.add_argument("--sub-position-vertical", type=float, default=1.0, help="Determine the specific vertical location of the subtitles")
parser.add_argument("--sub-position-horizontal", type=float, default=0.5, help="Determine the specific horizontal location of the subtitles")
parser.add_argument("--sub-alignment", choices=["left", "mid", "right"], default="mid", help="Determine the specific alignment of the subtitles")
parser.add_argument("--sub-color", choices=["white","yellow"], default="yellow", help="Determine the specific color of the subtitles")

args = parser.parse_args()


# Create video
create_video(
    args.input,
    fps=args.fps,
    video_dim_w=args.video_dim_w,
    video_dim_h=args.video_dim_h,
    speed=args.speed,
    add_sub=args.add_sub,
    font_type=args.font_type,
    sub_position_vertical=args.sub_position_vertical,
    sub_position_horizontal=args.sub_position_horizontal,
    sub_alignment=args.sub_alignment,
    sub_color=args.sub_color
)

print(f"Result is saved in {args.input}")
