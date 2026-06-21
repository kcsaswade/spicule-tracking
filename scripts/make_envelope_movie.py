from pathlib import Path
import cv2

FRAME_DIR = Path("outputs/50G/envelope_frames")
OUTFILE = "outputs/50G/movies/envelope_detection.mp4"

FPS = 10


def main():

    files = sorted(FRAME_DIR.glob("frame_*.png"))

    first = cv2.imread(str(files[0]))
    height, width, _ = first.shape

    writer = cv2.VideoWriter(
        OUTFILE,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )

    for f in files:
        img = cv2.imread(str(f))
        writer.write(img)

    writer.release()

    print(f"Saved movie to {OUTFILE}")


if __name__ == "__main__":
    main()