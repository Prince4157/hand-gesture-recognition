import cv2
import numpy as np
from tensorflow.keras.models import load_model


def run_live_demo():
    model = load_model('best_real_model.keras')
    classes = ['ROCK', 'PAPER', 'SCISSORS']
    cap = cv2.VideoCapture(0)

    print("Tips for Accuracy:")
    print("- Ensure your hand is well-lit")
    print("- Use a plain background (white wall is best)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Center Box
        x1, y1, x2, y2 = w // 2 - 120, h // 2 - 120, w // 2 + 120, h // 2 + 120
        roi = frame[y1:y2, x1:x2]

        # PREPROCESSING (Crucial)
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # PREDICT
        prediction = model.predict(img, verbose=0)
        idx = np.argmax(prediction[0])
        confidence = prediction[0][idx]

        # UI
        color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)
        label = f"{classes[idx]} {confidence * 100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Final Project - Gesture Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo()