import os
import cv2
from ultralytics import YOLO


class Image:
    def __init__(self, image_path = None):
        self.path = image_path
        self.data = cv2.imread(image_path)

        if self.data is None:
            raise ValueError(f"Failed to load image: {image_path}")

    def resize(self, scale):
        width = self.data.shape[1] * scale
        height = self.data.shape[0] * scale
        self.data = cv2.resize(self.data, (width, height), interpolation=cv2.INTER_CUBIC)

    def to_grayscale(self):
        return cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

    def set_data(self, new_data):
        self.data = new_data


class Preprocessor:
    def __init__(self, scale=8):
        self.scale = scale

    def process(self, image: Image):
        image.resize(self.scale)

        gray = image.to_grayscale()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharp = cv2.addWeighted(gray, 2.0, blur, -1.0, 0)

        sharp_color = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
        image.set_data(sharp_color)
        return image


class YOLODetector:
    def __init__(self, model_path, classes = None, threshold = 0.001):
        self.model = YOLO(model_path)
        self.classes = classes if classes else []
        self.threshold = threshold

    def detect_image(self, image: Image):
        results = self.model.predict(image.data, conf = self.threshold, verbose = False)[0]

        count = 0
        for box in results.boxes:
            class_index = int(box.cls[0])
            label = self.model.names[class_index]
            if label in self.classes:
                count += 1
        return count


class BirdDetector:
    def __init__(self, folder_path, model_path, classes = None, scale = 8, threshold = 0.001):
        self.folder = folder_path
        self.preprocessor = Preprocessor(scale = scale)
        self.detector = YOLODetector(model_path, classes = classes, threshold= threshold)

    def run(self):
        for file_name in os.listdir(self.folder):
            if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(self.folder, file_name)
            image = Image(image_path = image_path)
            processed = self.preprocessor.process(image)

            count = self.detector.detect_image(processed)
            print(f"{file_name}: {count}")


pipeline = BirdDetector(
    folder_path="bird_miniatures",
    model_path="yolov8s.pt",
    classes=["bird", "kite", "airplane"],
    scale=8,
    threshold=0.001
)

pipeline.run()