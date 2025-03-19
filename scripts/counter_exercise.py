import sys
import os
import glob
import cv2
import numpy as np
import torch
import json
from ultralytics import YOLO
from collections import deque
from src.models.heatmap_fpn_v3 import heatmap_fpn
from src.utils.heatmaps import extract_keypoints_with_confidence, extract_bbox_from_heatmaps
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Device configuration
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print(device)
#dtype = torch.float16 if torch.cuda.is_available() else torch.float32

FRAME_SKIP = 5  # Process every 3rd frame for efficiency
CONF_HISTORY = 5  # Number of past classifications to consider

def calculate_angle(a: list, b: list, c: list) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def load_model(our_model = True):
    if our_model:
        model = heatmap_fpn(num_classes=20, num_keypoints=17, backbone='resnet50').to(device)
        state_dict = torch.load('model_epoch_130.pth', map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        return YOLO('yolov8x-pose.pt')

class ContextAwareExerciseCounter:
    def __init__(self, conf_threshold: float = 0.3, class_stability: int = 5):
        self.counter = 0
        self.stage = None
        self.conf_threshold = conf_threshold
        self.previous_classes = deque(maxlen=class_stability)
        self.current_class = None
        self.last_exercise_type = None
        self.last_stage = None
        self.last_counter = 0

    def update_class(self, detected_class: str, confidence: float, conf_thresh: float = 0.5):
        if confidence > conf_thresh:
            self.previous_classes.append(detected_class)
            most_common = max(set(self.previous_classes), key=self.previous_classes.count)
            self.current_class = most_common

    def _get_angle(self, keypoints, kp_confs, indices):
        if all(i < len(keypoints) and kp_confs[i] >= self.conf_threshold for i in indices):
            return calculate_angle(keypoints[indices[0]], keypoints[indices[1]], keypoints[indices[2]])
        return None

    def process_frame(self, frame, keypoints, kp_confs, bbox, exercise_type, orig_h, orig_w, is_skipped=False):
        if is_skipped and self.last_exercise_type is not None:
            exercise_type = self.last_exercise_type
            self.stage = self.last_stage
            self.counter = self.last_counter
        else:
            left_indices, right_indices, key_body_part = {
                "deadlift": ((11, 13, 15), (12, 14, 16), "whole body"),
                "squat": ((11, 13, 15), (12, 14, 16), "lower body"),
                "push-up": ((5, 7, 9), (6, 8, 10), "upper body"),
                "benchpress": ((5, 7, 9), (6, 8, 10), "upper body")
            }.get(exercise_type, (None, None, None))

            if not left_indices:
                return frame

            left_angle = self._get_angle(keypoints, kp_confs, left_indices)
            right_angle = self._get_angle(keypoints, kp_confs, right_indices)
            angle = (left_angle + right_angle) / 2.0 if left_angle and right_angle else left_angle or right_angle
            if angle is None:
                return frame

            thresholds = {
                "upper body": {"start": 120, "end": 100},
                "lower body": {"start": 150, "end": 100},
                "whole body": {"start": 130, "end": 110}
            }

            if key_body_part not in thresholds:
                return frame

            start_th, end_th = thresholds[key_body_part]["start"], thresholds[key_body_part]["end"]

            new_stage = "start" if angle >= start_th else "end" if angle <= end_th else self.stage

            if self.stage == "end" and new_stage == "start":
                self.counter += 1

            self.stage = new_stage
            self.last_exercise_type = exercise_type
            self.last_stage = self.stage
            self.last_counter = self.counter

        # Draw keypoints
        for x, y, _ in keypoints:
            x = int(x * orig_w)
            y = int(y * orig_h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw skeleton
        skeleton_pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

        for i, j in skeleton_pairs:
            if i >= len(keypoints) or j >= len(keypoints) or keypoints[i][0] == 0 or keypoints[i][1] == 0 or keypoints[j][0] == 0 or keypoints[j][1] == 0:
                print('skipping because out of frame')
                continue  # Skip if any keypoint is out of frame
            if i < len(keypoints) and j < len(keypoints):
                xi, yi = keypoints[i][0], keypoints[i][1]
                xj, yj = keypoints[j][0], keypoints[j][1]
                xi, yi = int(xi * orig_w), int(yi * orig_h)
                xj, yj = int(xj * orig_w), int(yj * orig_h)
                cv2.line(frame, (xi, yi), (xj, yj), (255, 0, 0), 2)

        # Ensure bbox is properly converted to a list
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy().tolist()[0]
        # Draw bounding box
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = int(x1 * orig_w), int(y1 * orig_h)
            x2, y2 = int(x2 * orig_w), int(y2 * orig_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw exercise label
        cv2.putText(frame, f'Exercise: {exercise_type}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        cv2.putText(frame, f'Reps: {self.counter} | Stage: {self.stage}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return frame

def frames_to_video(frames_dir, output_filename="output_video.mp4", fps=30, frame_size=(352, 352)):
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        print("No frames found. Video creation aborted.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            continue  # Skip invalid frames

        frame_resized = cv2.resize(frame, frame_size)  # Resize to match prediction size
        out.write(frame_resized)

    out.release()
    print(f"Video saved successfully: {output_filename}")


def predict(frame, model, fallback_model, class_smoothing):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    target_w, target_h = [352, 352]

    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h))

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    sample_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    sample_tensor = (sample_tensor - mean) / std
    sample_tensor = sample_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sample_tensor)

    bbox_pred = extract_bbox_from_heatmaps(output[0])
    keypoints_pred = extract_keypoints_with_confidence(output[1], refine=False)

    keypoints_conf = [kp[2] for kp in keypoints_pred[0] if len(kp) > 2]
    if len(keypoints_conf) == 0 or np.mean(keypoints_conf) < 0.3:
        yolo_result = fallback_model(frame)
        if yolo_result[0].keypoints is not None:
            keypoints_pred = yolo_result[0].keypoints.xy.cpu().numpy()
            bbox_pred = yolo_result[0].boxes.xyxy.cpu().numpy()[0] if yolo_result[0].boxes else None

    with open('heatmap_fpn_v3.json', 'r') as f:
        idx_to_class_name = json.load(f)
    probabilities = torch.softmax(output[2][0], dim=0)
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class_name = idx_to_class_name[str(predicted_class_idx)]
    class_smoothing.append(predicted_class_name)
    smoothed_class = max(set(class_smoothing), key=class_smoothing.count)

    return bbox_pred, keypoints_pred, smoothed_class, h, w

def main(our_model = True):
    video_path = f"notebooks/video_samples/squat_sample.mp4"
    output_frames_dir = "demo/output_frames/"
    output_video_path = f"demo/squat_processed_new.mp4"
    os.makedirs(output_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    counter = ContextAwareExerciseCounter()
    class_smoothing = deque(maxlen=CONF_HISTORY)
    frame_index = 0
    model = load_model(our_model)
    fallback_model = YOLO('yolov8x-pose.pt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_index % FRAME_SKIP != 0:
            continue

        if our_model:
            bbox, keypoints, exercise, h, w = predict(frame, model, fallback_model, class_smoothing)
            kp_confs = []
            for i, point in enumerate(keypoints[0]):
                x, y, confidence = point
                kp_confs.append(confidence)
            frame = counter.process_frame(frame, keypoints[0], kp_confs, bbox, exercise, h, w)

        else:
            model = load_model(our_model)
            result = model(frame)
            # currenlty only processing in yolo format frame
            if result[0].keypoints is not None and len(result[0].keypoints.xy.cpu().numpy()[0]) >= 17:
                keypoints = result[0].keypoints.xy.cpu().numpy()[0]
                kp_confs = result[0].keypoints.conf.cpu().numpy()[0]
                bbox = result[0].boxes.xyxy.cpu().numpy()[0] if result[0].boxes else None
                frame = counter.process_frame(frame, keypoints, kp_confs, bbox, exercise)

        frame_filename = os.path.join(output_frames_dir, f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_index += FRAME_SKIP

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final {exercise} count: {counter.counter}")
    frames_to_video(output_frames_dir, output_video_path, fps / FRAME_SKIP)

if __name__ == "__main__":
    IMAGE_DIR = "demo/output_frames/"
    for file in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
        os.remove(file)
    main(our_model=True)
