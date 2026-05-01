import json
import cv2
from pathlib import Path
from ultralytics import YOLO


class Image:
    def __init__(self, image_path=None, frame=None):
        self.path = image_path

        if frame is not None:
            self.data = frame
        elif image_path is not None:
            self.data = cv2.imread(image_path)
        else:
            raise ValueError("Image needs either image_path or frame")

    def add_rectangle(self, bounding_box, color=(0, 255, 0), thickness=2):
        cv2.rectangle(
            self.data,
            (bounding_box.start_x, bounding_box.start_y),
            (bounding_box.end_x, bounding_box.end_y),
            color,
            thickness
        )

    def add_text(self, text, x, y, color=(0, 255, 0)):
        cv2.putText(
            self.data,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    def draw_detection(self, detection):
        bounding_box = detection.bounding_box
        self.add_rectangle(bounding_box)
        self.add_text(
            f"{detection.name} {detection.confidence:.2f}",
            bounding_box.start_x,
            bounding_box.start_y - 5
        )

    def draw_detections(self, detections):
        for detection in detections.array:
            self.draw_detection(detection)

    def save(self, output_path):
        cv2.imwrite(output_path, self.data)


class Video:
    def __init__(self, video_path):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.writer = None

    def create_writer(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def read_frame(self):
        return self.cap.read()

    def write_frame(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        self.cap.release()
        if self.writer is not None:
            self.writer.release()


class BoundingBox:
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

    def to_dictionary(self):
        return {
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y
        }


class Detection:
    def __init__(self, name, confidence, bounding_box):
        self.name = name
        self.confidence = float(confidence)
        self.bounding_box = bounding_box

    def to_dictionary(self):
        return {
            "class": self.name,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dictionary()
        }


class Detections:
    def __init__(self):
        self.array = []

    def add_detection(self, detection):
        self.array.append(detection)

    def clear(self):
        self.array = []

    def to_json_ready(self):
        return [detection.to_dictionary() for detection in self.array]


class JSONFile:
    def __init__(self, output_path):
        self.path = output_path

    def save(self, data):
        with open(self.path, "w") as file:
            json.dump(data, file, indent = 4)


class ImageModel:
    def __init__(self, model_name):
        self.name = model_name
        self.base = YOLO(model_name)
        self.detections = Detections()

    def print_info(self):
        print(f"Model loaded: {self.name}")
        print(f"Number of classes: {len(self.base.names)}")
        print(f"Classes: {self.base.names}")

    def detect_image(self, image_path, save_path="output.png", confidence_threshold=0.3):
        self.detections.clear()

        image = Image(image_path = image_path)
        results = self.base(image_path)[0]

        for box_boundaries, box_class_index, box_confidence in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):

            if box_confidence >= confidence_threshold:

                start_x, start_y, end_x, end_y = map(int, box_boundaries)
                class_name = self.base.names[int(box_class_index)]

                bounding_box = BoundingBox(start_x, start_y, end_x, end_y)
                detection = Detection(class_name, box_confidence, bounding_box)

                self.detections.add_detection(detection)

        image.draw_detections(self.detections)
        image.save(save_path)

        json_path = Path(save_path).with_name(f"{Path(save_path).stem}_confidence.json")

        json_file = JSONFile(json_path)
        json_file.save(self.detections.to_json_ready())

        print(f"Saved annotated image to {save_path}")
        print(f"Saved JSON to {json_path}")

        return self.detections.array


class VideoModel:
    def __init__(self, model_name):
        self.name = model_name
        self.base = YOLO(model_name)

    def detect_video(self, video_path,
                     video_output_path = "output_video.mp4",
                     json_output_path = "video_detections.json",
                     confidence_threshold = 0.3):

        video = Video(video_path)
        video.create_writer(video_output_path)

        full_video_structure = {
            "video_path": video_path,
            "frames": []
        }

        frame_index = 0

        while True:
            returned, frame = video.read_frame()
            if not returned:
                break

            results = self.base.predict(frame, verbose=False)[0]
            detections = Detections()

            for box_boundaries, box_class_index, box_confidence in zip(
                    results.boxes.xyxy, results.boxes.cls, results.boxes.conf):

                if box_confidence >= confidence_threshold:

                    start_x, start_y, end_x, end_y = map(int, box_boundaries)
                    class_name = self.base.names[int(box_class_index)]

                    bounding_box = BoundingBox(start_x, start_y, end_x, end_y)
                    detection = Detection(class_name, float(box_confidence), bounding_box)
                    detections.add_detection(detection)

            image = Image(frame = frame)
            image.data = frame
            image.draw_detections(detections)

            video.write_frame(image.data)

            full_video_structure["frames"].append({
                "frame_index": frame_index,
                "detections": detections.to_json_ready()
            })

            frame_index += 1

        video.release()

        json_file = JSONFile(json_output_path)
        json_file.save(full_video_structure)

        print(f"Saved processed video to {video_output_path}")
        print(f"Saved JSON to {json_output_path}")

        return full_video_structure


def detect_image_with_thresholds(model, image_path, thresholds = [0.1, 0.3, 0.5, 0.7]):
    results = {}

    for threshold in thresholds:
        print(f"\n--- Detection for threshold {threshold} ---")

        save_path = f"output_conf_{threshold}.png"
        detections = model.detect_image(image_path, save_path, confidence_threshold = threshold)
        results[threshold] = detections

    return results


def detect_video_with_thresholds(model, video_path, video_name, thresholds=[0.1, 0.3, 0.5, 0.7]):
    all_stats = {}

    for threshold in thresholds:
        print(f"\n--- Detection for threshold {threshold} ---")

        output_video = f"output_video_conf_{threshold}_{video_name}.mp4"
        json_output = f"video_detections_conf_{threshold}_{video_name}.json"

        full_structure = model.detect_video(
            video_path = video_path,
            video_output_path = output_video,
            json_output_path = json_output,
            confidence_threshold = threshold
        )

        class_counter = {}
        for frame in full_structure["frames"]:
            for detection in frame["detections"]:
                class_index = detection["class"]
                class_counter[class_index] = class_counter.get(class_index, 0) + 1

        all_stats[threshold] = class_counter

        stats_name = f"stats_conf_{threshold}.json"
        JSONFile(stats_name).save(class_counter)
        print(f"Stats saved to: {stats_name}")

    return all_stats


def print_stats(stats):
    for threshold, classes in stats.items():
        print(f"\n=== Stats for threshold {threshold} ===")
        total = sum(classes.values())
        print(f"Sum detections: {total}")
        for class_name, count in classes.items():
            print(f"{class_name} : {count}")


image_model = ImageModel("yolov8n.pt")
image_model.print_info()

detect_image_with_thresholds(image_model, "office_yolo.png")

video_model = VideoModel("yolov8n.pt")
stats = detect_video_with_thresholds(video_model, "street_yolo.mp4", "street")
print_stats(stats)

stats2 = detect_video_with_thresholds(video_model, "office_yolo.mp4", "office")
print_stats(stats2)
