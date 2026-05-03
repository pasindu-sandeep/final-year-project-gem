import cv2
import time
from pathlib import Path


def capture_angle_image(angle, camera_index=1, save_dir="captures"):
    """
    Capture one image from USB webcam and save it as angle_<angle>.jpg
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Camera index {camera_index} not detected.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Let camera wake/exposure settle
    time.sleep(1)

    frame = None
    ret = False

    # Read multiple frames to avoid old/dark frame
    for _ in range(15):
        ret, frame = cap.read()
        time.sleep(0.05)

    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to capture image at {angle} degrees.")

    image_path = save_dir / f"angle_{angle}.jpg"

    saved = cv2.imwrite(str(image_path), frame)

    if not saved or not image_path.exists():
        raise RuntimeError(f"Image file was not saved: {image_path}")

    return str(image_path)