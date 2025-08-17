import os
import cv2
import json
import tempfile
import pandas as pd
from ultralytics import YOLO
from yt_dlp import YoutubeDL

# Vehicle classes from COCO dataset: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# Download YouTube video
def download_video(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        'format': 'mp4/bestvideo+bestaudio/best',
        'outtmpl': os.path.join(out_dir, 'traffic.mp4'),
        'merge_output_format': 'mp4',
        'quiet': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return os.path.join(out_dir, 'traffic.mp4')

# Assign lane by dividing frame into 3 vertical sections
def get_lane(cx, frame_width):
    third = frame_width / 3
    if cx < third:
        return 1
    elif cx < 2 * third:
        return 2
    else:
        return 3

# Main function
def main():
    video_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
    temp_dir = tempfile.mkdtemp()
    video_path = download_video(video_url, temp_dir)

    # Load YOLO model (pre-trained on COCO)
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("output_annotated.mp4", fourcc, fps, (width, height))

    lane_counts = {1: 0, 2: 0, 3: 0}
    counted_ids = set()
    records = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model.track(frame, persist=True, conf=0.25, tracker="bytetrack.yaml")
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, ids, clss):
                if cls not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                lane = get_lane(cx, width)

                # Count vehicle once when crossing 60% of height
                if cy > int(height * 0.6) and track_id not in counted_ids:
                    lane_counts[lane] += 1
                    counted_ids.add(track_id)
                    timestamp = round(frame_num / fps, 2)
                    records.append({
                        "vehicle_id": track_id,
                        "lane": lane,
                        "frame": frame_num,
                        "timestamp_seconds": timestamp
                    })

                # Draw box and lane info
                color = (0, 255, 0) if lane == 1 else (255, 0, 0) if lane == 2 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} L{lane}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw lane dividers
        third = width // 3
        cv2.line(frame, (third, 0), (third, height), (255, 255, 255), 2)
        cv2.line(frame, (2 * third, 0), (2 * third, height), (255, 255, 255), 2)

        # Show live counts
        cv2.putText(frame, f"Lane 1: {lane_counts[1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Lane 2: {lane_counts[2]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Lane 3: {lane_counts[3]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_video.write(frame)

    cap.release()
    out_video.release()

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv("vehicle_counts.csv", index=False)

    print("Processing complete.")
    print("Lane counts:", lane_counts)
    print("Output video: output_annotated.mp4")
    print("CSV file: vehicle_counts.csv")

if __name__ == "__main__":
    main()