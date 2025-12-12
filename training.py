from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import cv2
import math
import os

class ShapeTrainer:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.drawing = False
        self.points = []
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        self.X_train = []  # Features
        self.y_train = []  # Labels
        self.model = None
        self.current_label = None

        cv2.namedWindow('Shape Trainer')
        cv2.setMouseCallback('Shape Trainer', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points = [(x, y)]
            self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.points.append((x, y))
                if len(self.points) > 1:
                    cv2.line(self.canvas, self.points[-2], self.points[-1],
                             (0, 0, 0), 3)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.points) > 5:
                self.save_training_example()

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

    def save_training_example(self):
        if self.current_label is None:
            print("No label set! Press 1-5 to set label first.")
            return

        features = self.calculate_features()

        if features is None:
            print("Invalid drawing, not saved")
            return

        feature_array = [
            features['aspect_ratio'],
            features['num_corners'],
            features['circularity'],
            features['extent'],
            1 if features['is_closed'] else 0
        ]

        self.X_train.append(feature_array)
        self.y_train.append(self.current_label)

        print(f"Saved to {self.current_label} (total: {len(self.X_train)} examples)")

    def train_model(self):
        if len(self.X_train) < 10:
            print(f"Need at least 10 examples. Currently have: {len(self.X_train)}")
            return

        unique_labels = set(self.y_train)
        if len(unique_labels) < 2:
            print(f"Need at least 2 different shapes. Currently have: {unique_labels}")
            return

        print("\n" + "=" * 50)
        print("Training model...")
        print("=" * 50)

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        from collections import Counter
        class_counts = Counter(y)
        print("\nClass distribution:")
        for label, count in class_counts.items():
            print(f"  {label}: {count} examples")

        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y
            print("\nSmall dataset - using all data for both train and test")

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)

        print(f"\nModel trained!")
        print(f"   Train Accuracy: {train_accuracy * 100:.1f}%")
        print(f"   Test Accuracy: {test_accuracy * 100:.1f}%")
        print("=" * 50 + "\n")

        with open('shape_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved to 'shape_model.pkl'")

    def draw_ui(self):
        """Draw user interface"""
        display = self.canvas.copy()

        # Top bar
        cv2.rectangle(display, (0, 0), (self.width, 100), (50, 50, 50), -1)

        # Title
        cv2.putText(display, "TRAINING MODE", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Current label
        if self.current_label:
            label_text = f"Label: {self.current_label}"
            cv2.putText(display, label_text, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(display, "Press 1-5 to set label", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Bottom instructions
        y_pos = self.height - 150

        instructions = [
            "CONTROLS:",
            "1 - CIRCLE  |  2 - SQUARE  |  3 - TRIANGLE",
            "4 - LINE    |  5 - STAR",
            "T - Train model  |  C - Clear  |  Q - Quit",
            f"Collected: {len(self.X_train)} examples"
        ]

        for i, text in enumerate(instructions):
            cv2.putText(display, text, (10, y_pos + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return display

    def run(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("SHAPE TRAINER - TRAINING MODE ONLY")
        print("=" * 60)
        print("\nCONTROLS:")
        print("  1 - Label as CIRCLE")
        print("  2 - Label as SQUARE")
        print("  3 - Label as TRIANGLE")
        print("  4 - Label as LINE")
        print("  5 - Label as STAR")
        print("  T - Train model on collected examples")
        print("  C - Clear canvas")
        print("  Q - Quit")
        print("\nWorkflow:")
        print("  1. Press 1-5 to set label")
        print("  2. Draw shape with mouse")
        print("  3. Shape saves automatically")
        print("  4. Repeat for 10+ examples per shape")
        print("  5. Press T to train model")
        print("=" * 60 + "\n")

        label_map = {
            ord('1'): 'CIRCLE',
            ord('2'): 'SQUARE',
            ord('3'): 'TRIANGLE',
            ord('4'): 'LINE',
            ord('5'): 'STAR'
        }

        while True:
            display = self.draw_ui()
            cv2.imshow('Shape Trainer', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                self.canvas = np.ones((self.height, self.width, 3),
                                      dtype=np.uint8) * 255
                self.points = []

            elif key == ord('t'):
                self.train_model()

            elif key in label_map:
                self.current_label = label_map[key]
                print(f"\n🏷Label set to: {self.current_label}")

        cv2.destroyAllWindows()

        # Save raw training data on exit
        if len(self.X_train) > 0:
            data = {
                'X': self.X_train,
                'y': self.y_train
            }
            with open('training_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            print("\nTraining data saved to 'training_data.pkl'")


if __name__ == "__main__":
    trainer = ShapeTrainer()
    trainer.run()