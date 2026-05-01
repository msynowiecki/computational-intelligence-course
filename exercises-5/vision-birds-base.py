import os
import cv2


class Image:
    def __init__(self, image_path = None):
        self.path = image_path
        self.data = cv2.imread(image_path)

        if self.data is None:
            raise ValueError(f"Failed to load image: {image_path}")

    def to_grayscale(self):
        return cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

    def save(self, output_path):
        cv2.imwrite(output_path, self.data)


class BinaryImage:
    def __init__(self, gray_image, threshold=128):
        self.data = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    def find_contours(self):
        contours, _ = cv2.findContours(self.data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


class BirdCounter:
    def __init__(self, folder_path):
        self.folder = folder_path
        self.results = []

    def load_images(self):
        supported = ('.png', '.jpg', '.jpeg')
        files = [file for file in os.listdir(self.folder) if file.endswith(supported)]
        return [os.path.join(self.folder, file) for file in files], files

    def count_birds(self):
        paths, filenames = self.load_images()

        for path, name in zip(paths, filenames):
            image = Image(image_path=path)
            gray = image.to_grayscale()

            binary_image = BinaryImage(gray)
            contours = binary_image.find_contours()

            count = len(contours)

            self.results.append({
                "file": name,
                "birds": count
            })

        return self.results

    def print_results(self):
        for result in self.results:
            print(f"{result['file']} : {result['birds']}")

counter = BirdCounter("bird_miniatures")
counter.count_birds()
counter.print_results()