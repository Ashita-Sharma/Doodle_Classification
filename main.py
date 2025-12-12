from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import cv2
import math
import os


class MLShapeRecognizer:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.drawing = False
        self.points = []
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        self.recognized_shape = None
        self.confidence = 0

        self.X_train = []  # Features
        self.y_train = []  # Labels
        self.model = None

        self.mode = 'testing'
        self.current_label = None

        cv2.namedWindow('ML Shape Recognizer')
        cv2.setMouseCallback('ML Shape Recognizer', self.mouse_callback)
        self.load_model()

    def load_model(self):
        if os.path.exists('shape_model.pkl'):
            with open('shape_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("Loaded existing model!")
        else:
            print("No model found. Switch to training.py to create one.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.points = [(x, y)]
            self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            self.recognized_shape = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Add point while drawing
                self.points.append((x, y))
                # Draw line from previous point
                if len(self.points) > 1:
                    cv2.line(self.canvas, self.points[-2], self.points[-1],
                             (0, 0, 0), 3)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            self.drawing = False
            if len(self.points) > 5:  # Need minimum points
                if self.mode == 'testing':
                    self.recognize_shape()

    def calculate_features(self):
        if len(self.points) < 5:
            return None

        points_array = np.array(self.points, dtype=np.float32)

        # 1. Bounding box
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        width = max_x - min_x
        height = max_y - min_y

        if width == 0 or height == 0:
            return None

        # 2. Aspect ratio
        aspect_ratio = width / height

        # 3. Is path closed?
        start_point = self.points[0]
        end_point = self.points[-1]
        distance = math.sqrt((end_point[0] - start_point[0]) ** 2 +
                             (end_point[1] - start_point[1]) ** 2)
        is_closed = distance < 30

        # 4. Approximate polygon (find corners)
        epsilon = 0.02 * cv2.arcLength(points_array, is_closed)
        approx = cv2.approxPolyDP(points_array, epsilon, is_closed)
        num_corners = len(approx)

        # 5. Circularity
        area = cv2.contourArea(points_array)
        perimeter = cv2.arcLength(points_array, is_closed)

        if perimeter == 0:
            circularity = 0
        else:
            circularity = (4 * math.pi * area) / (perimeter ** 2)

        # 6. Extent
        bbox_area = width * height
        extent = area / bbox_area if bbox_area > 0 else 0

        features = {
            'aspect_ratio': aspect_ratio,
            'num_corners': num_corners,
            'circularity': circularity,
            'extent': extent,
            'is_closed': is_closed
        }

        return features

    def recognize_shape(self):
        if self.model is None:
            self.recognized_shape = "No model trained!"
            self.confidence = 0
            print("No model loaded. Switch to training.py mode first!")
            return

        features = self.calculate_features()

        if features is None:
            self.recognized_shape = "Invalid drawing"
            self.confidence = 0
            return

        feature_array = [
            features['aspect_ratio'],
            features['num_corners'],
            features['circularity'],
            features['extent'],
            1 if features['is_closed'] else 0
        ]
        try:
            prediction = self.model.predict([feature_array])
            probabilities = self.model.predict_proba([feature_array])

            self.recognized_shape = prediction[0]
            self.confidence = np.max(probabilities) * 100

            print(f"Recognized: {self.recognized_shape} ({self.confidence:.1f}%)")
        except Exception as e:
            self.recognized_shape = "Error"
            self.confidence = 0
            print(f"Prediction error: {e}")


    def draw_ui(self):
        display = self.canvas.copy()

        # Top bar
        cv2.rectangle(display, (0, 0), (self.width, 100), (50, 50, 50), -1)

        # Mode indicator
        if self.mode == 'testing':
            mode_text = "MODE: TESTING"
            mode_color = (0, 255, 0)

            if self.recognized_shape:
                result_text = f"{self.recognized_shape}"
                conf_text = f"{self.confidence:.1f}%"

                color = (0, 255, 0) if self.confidence > 70 else (0, 165, 255)
                cv2.putText(display, result_text, (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(display, conf_text, (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(display, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

        # Bottom instructions
        y_pos = self.height - 120


        instructions = [
                "TESTING MODE:",
                "Draw a shape to recognize it",
                "C: Clear"
            ]

        for i, text in enumerate(instructions):
            cv2.putText(display, text, (10, y_pos + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return display

    def run(self):

        print("\n" + "=" * 60)
        print("ML SHAPE RECOGNIZER")
        print("=" * 60)
        print("\nCONTROLS:")
        print("  C - Clear canvas")
        print("  Q - Quit")
        print("\nTESTING MODE:")
        print("  Draw shape to see prediction")
        print("=" * 60 + "\n")

        # Label mappings
        label_map = {
            ord('1'): 'CIRCLE',
            ord('2'): 'SQUARE',
            ord('3'): 'TRIANGLE',
            ord('4'): 'LINE',
            ord('5'): 'STAR'
        }

        while True:
            display = self.draw_ui()
            cv2.imshow('ML Shape Recognizer', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break


            elif key == ord('c'):
                # Clear canvas
                self.canvas = np.ones((self.height, self.width, 3),
                                      dtype=np.uint8) * 255
                self.points = []
                self.recognized_shape = None


        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = MLShapeRecognizer()
    recognizer.run()