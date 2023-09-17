import cv2
import numpy as np
import pyvirtualcam
import time
import mediapipe as mp
import tensorflow as tf
import json
import os

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

# Initialize virtual webcam. Set the width, height, and FPS to match source webcam.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
i = 0 # Counter for file naming

# Initialize mediapipe hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the code and create a video writer object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1'
# out = None
# start_time = None
# recording = False

start_time = time.time()
hand_detected = False
time_hand_detected = None
pred = ""

#Load Model
loaded_model = tf.keras.models.load_model('multi_layer_perceptron.h5')

def save_landmarks(landmarks):
    landmarks_data = []
    for hand_landmarks in landmarks:
        hand_data = []
        for point in hand_landmarks.landmark:
            hand_data.append({"x": point.x, "y": point.y, "z": point.z})
        landmarks_data.append(hand_data)

    return json.dumps(landmarks_data)

def make_prediction(snap):
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        
        image = cv2.flip(cv2.imread(snap), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # if results.multi_hand_landmarks is None:
        #     return "No hand detected"
        
        snap_json = save_landmarks(results.multi_hand_landmarks)  # Assuming save_landmarks returns a JSON string
        landmarks_data = json.loads(snap_json)  # Convert JSON string back to Python object

        # Convert landmarks_data into a flat feature vector suitable for the model
        feature_vector = []
        for hand_data in landmarks_data:
            for point in hand_data:
                feature_vector.extend([point['x'], point['y'], point['z']])
                
        feature_vector = np.array([feature_vector])  # Convert list to numpy array and reshape for prediction
        
        predictions = loaded_model.predict(feature_vector)
        labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V',
                'W', 'X', 'Y', 'Z']
        
        max_index = np.argmax(predictions)
        predicted_label = labels[max_index]
        return predicted_label

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        while True:
            
            # Capture frame-by-frame from the real webcam
            ret, image = cap.read()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:

                if time_hand_detected is None:
                    time_hand_detected = time.time()  # Record the time the hand was detected
                                
                elapsed_time = time.time() - time_hand_detected
                if elapsed_time >= 1.0:
                    # Save the snapshot
                    cv2.imwrite(f'hand_snapshot{i}.png', image)
                    time_hand_detected = None  # Reset the timer after taking the snapshot
                    # Remove the break statement if you want to continue capturing

                    pred = make_prediction(f'hand_snapshot{i}.png')
                    print(pred)

                    i += 1

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            else:
                time_hand_detected = None  # Reset the timer if no hand is detected
            
            # Calculate text size and position to display it at the center bottom
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 6
            font_thickness = 16
            text_size = cv2.getTextSize(pred, font, font_size, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50  # 50 pixels from the bottom

            # Add white highlights (by drawing the text with a thicker white line)
            cv2.putText(image, pred, (text_x, text_y), font, font_size, (255,255,255), font_thickness+4, cv2.LINE_AA)

            # Add text (black font)
            cv2.putText(image, pred, (text_x, text_y), font, font_size, (0,0,0), font_thickness, cv2.LINE_AA)

            # Flip frame 
            text_frame = cv2.flip(image, 1)

            resized_frame = cv2.resize(text_frame, (cam.width, cam.height))
            flipped_frame = cv2.flip(resized_frame, 1)

            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Send to virtual camera
            cam.send(frame_rgb)

            # Optional: Sleep for a while to sync FPS (use if needed)
            cam.sleep_until_next_frame()

            # Display the resulting frame for debugging
            #cv2.imshow('Real Camera Output', frame)

            # Press 'q' to quit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# if out is not None:
#     out.release()

cap.release()
cv2.destroyAllWindows()