import os
import cv2
import tensorflow as tf
import numpy as np
import dlib
import gdown

# shape_predictor_68_face_landmarks.dat
url1 = 'https://drive.google.com/uc?id=19lnP3OFf1mQUdWMy9Nlmt2hyXJmy707z'
output1 = 'shape_predictor_68_face_landmarks.dat'
gdown.download(url1, output1, quiet=False)

# model.keras
url2 = 'https://drive.google.com/uc?id=1dulEXU8N0BKwY93C22tp71GShmb4zVVp'
output2 = 'model.keras'
gdown.download(url2, output2, quiet=False)

# modelV2.keras
url3 = 'https://drive.google.com/uc?id=1TqMMDhDjJKuOUWpGRgg87xT6vWd_l1Ts'
output3 = 'model_V2.keras'
gdown.download(url3, output3, quiet=False)

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define vocabulary
vocab = list("abcdefghijklmnopqrstuvwxyz'?!123456789 ")
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# CTC Loss function
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Load trained model
def load_trained_model(model_path="model.keras"):
    return tf.keras.models.load_model(model_path, compile=True, custom_objects={"CTCLoss": CTCLoss})

# Process video and extract normalized lip region frames
def load_video(path, width=70, height=40, target_frames=75):
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = tf.image.rgb_to_grayscale(frame)
        faces = detector(frame.numpy())

        try:
            landmarks = predictor(frame.numpy(), faces[0])
            cx = (landmarks.part(48).x + landmarks.part(54).x) // 2
            cy = (landmarks.part(48).y + landmarks.part(54).y) // 2

            half_width = width // 2
            half_height = height // 2

            if (cy + half_height) > frame.shape[0]:
                cy = frame.shape[0] - half_height

            left = cx - half_width
            top = cy - half_height
            right = cx + half_width
            bottom = cy + half_height
            cropped = frame[top:bottom, left:right, :]
        except:
            cropped = frame[190:236, 80:220, :]

        frames.append(cropped)

    cap.release()
    frames = tf.convert_to_tensor(frames)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    norm_frames = tf.cast((frames - mean), tf.float32) / std

    num_frames = norm_frames.shape[0]

    if num_frames < target_frames:
        padding = tf.zeros([target_frames - num_frames, height, width, 1], dtype=tf.float32)
        norm_frames = tf.concat([norm_frames, padding], axis=0)
    elif num_frames > target_frames:
        norm_frames = norm_frames[:target_frames]

    return norm_frames


# Predict text from processed video
def predict_from_video(model, video):
    video = tf.expand_dims(video, axis=0)
    yhat = model.predict(video)

    decoded, _ = tf.keras.backend.ctc_decode(yhat, input_length=tf.constant([yhat.shape[1]]))
    decoded_text = tf.strings.reduce_join(num_to_char(decoded[0][0])).numpy().decode()

    return decoded_text 




# Wrapper function
def predict_lipreading(video_path, model_path="model.keras"):
    model = load_trained_model(model_path)
    frames = load_video(video_path)
    return predict_from_video(model, frames)




