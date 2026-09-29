"""
LIVE CAMERA DEMO - Works with Real Trained Model
Run this AFTER training the real model!
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

class LiveGestureDemo:
    def __init__(self, model_path='best_real_model.keras'):
        """Initialize live demo with real trained model"""

        try:
            self.model = load_model(model_path)
            print("✓ Real model loaded successfully!")
        except:
            print("❌ Error: Model not found!")
            print("Please train the model first by running the real dataset version")
            exit()

        self.class_names = ['ROCK', 'PAPER', 'SCISSORS']
        self.img_size = (150, 150)
        self.confidence_threshold = 60  # Lower threshold for real predictions

        # Prediction smoothing
        self.prediction_history = []
        self.history_size = 10

    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Resize
        resized = cv2.resize(frame, self.img_size)
        # Normalize
        normalized = resized.astype('float32') / 255.0
        # Add batch dimension
        input_img = np.expand_dims(normalized, axis=0)
        return input_img

    def get_stable_prediction(self, class_idx):
        """Smooth predictions over time"""
        self.prediction_history.append(class_idx)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        if len(self.prediction_history) >= 5:
            # Return most common prediction
            return max(set(self.prediction_history), key=self.prediction_history.count)
        return class_idx

    def run(self):
        """Run live gesture recognition"""

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return

        print("\n" + "="*60)
        print("LIVE HAND GESTURE RECOGNITION")
        print("Real Model - Actually Works!")
        print("="*60)
        print("\nInstructions:")
        print("  1. Show Rock (fist), Paper (flat hand), or Scissors")
        print("  2. Position hand in the GREEN BOX")
        print("  3. Hold steady for prediction")
        print("  4. Press 'Q' to quit")
        print("="*60 + "\n")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror view
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # Define ROI
            roi_size = min(400, min(width, height) - 100)
            x1 = (width - roi_size) // 2
            y1 = (height - roi_size) // 2 - 50
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            # Extract ROI
            roi = frame[y1:y2, x1:x2]

            # Only predict every 3 frames (faster)
            status_text = "SHOW GESTURE"
            status_color = (0, 165, 255)
            prediction_text = ""

            if frame_count % 3 == 0 and roi.size > 0:
                try:
                    # Preprocess and predict
                    input_img = self.preprocess_frame(roi)
                    prediction = self.model.predict(input_img, verbose=0)

                    class_idx = np.argmax(prediction)
                    confidence = prediction[0][class_idx] * 100

                    # Get stable prediction
                    stable_class = self.get_stable_prediction(class_idx)

                    if confidence >= self.confidence_threshold:
                        gesture = self.class_names[stable_class]
                        status_text = gesture
                        status_color = (0, 255, 0)  # Green
                        prediction_text = f"Confidence: {confidence:.1f}%"
                    else:
                        status_text = "HOLD STEADY..."
                        status_color = (0, 255, 255)  # Yellow

                    # Show probabilities
                    y_offset = height - 130

                    # Background for probabilities
                    cv2.rectangle(frame, (5, y_offset-30), (260, height-5),
                                (0, 0, 0), -1)

                    cv2.putText(frame, "Probabilities:", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    y_offset += 25

                    for i, name in enumerate(self.class_names):
                        prob = prediction[0][i] * 100
                        bar_width = int(prob * 1.5)

                        # Probability bar
                        cv2.rectangle(frame, (10, y_offset-15),
                                    (10+bar_width, y_offset-2),
                                    (0, 255, 0) if i == class_idx else (100, 100, 100), -1)

                        # Text
                        text = f"{name}: {prob:.0f}%"
                        color = (255,255,255) if i == class_idx else (150,150,150)
                        cv2.putText(frame, text, (10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y_offset += 25

                except Exception as e:
                    status_text = "PROCESSING..."

            # Draw ROI box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, "Place hand here", (x1+10, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Status background
            cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)

            # Main status
            cv2.putText(frame, status_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, status_color, 4)

            # Confidence
            if prediction_text:
                cv2.putText(frame, prediction_text, (20, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Instructions
            cv2.putText(frame, "Press 'Q' to quit", (width-200, height-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow('Hand Gesture Recognition - REAL MODEL', frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Demo closed!")


def main():
    import os

    print("\n" + "="*60)
    print("LIVE CAMERA DEMO - REAL DATASET MODEL")
    print("="*60)

    if not os.path.exists('best_real_model.keras'):
        print("\n❌ Model not found!")
        print("\nPlease train the real model first:")
        print("  python RealHandGesture.py")
        print("\nThen run this demo again.")
        return

    demo = LiveGestureDemo()
    demo.run()


if __name__ == "__main__":
    main()