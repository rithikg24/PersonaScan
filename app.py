import numpy as np
import dlib
import cv2
from PIL import Image
from imutils import face_utils
from colorthief import ColorThief
from deepface import DeepFace
import webcolors
from flask import Flask, request, jsonify
import os
from helper.btoi import base64_to_image

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    b64_string = data['b64_string']

    imgPath=base64_to_image(b64_string)

    attributes = {}

    # Reading the image and converting it into a numpy array
    image = dlib.load_rgb_image(imgPath)

    # Gender and Race Detection
    analysis_results = DeepFace.analyze(image)

    attributes['Gender'] = analysis_results[0]['dominant_gender']
    attributes['Race'] = analysis_results[0]['dominant_race']
    attributes['Emotion'] = analysis_results[0]['dominant_emotion']
    attributes['Age'] = analysis_results[0]['age']

    # Glasses Detection
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(r"Pre-trained Models\shape_predictor_68_face_landmarks.dat")

    face_rect = face_detector(image)[0]
    landmarks = landmark_predictor(image, face_rect)
    landmark_coords = np.array([[p.x, p.y] for p in landmarks.parts()])

    nose_bridge_x_coords = []
    nose_bridge_y_coords = []
    for i in [28, 29, 30, 31, 33, 34, 35]:
        nose_bridge_x_coords.append(landmark_coords[i][0])
        nose_bridge_y_coords.append(landmark_coords[i][1])

    x_min = min(nose_bridge_x_coords)
    x_max = max(nose_bridge_x_coords)
    y_min = landmark_coords[20][1]
    y_max = landmark_coords[31][1]

    cropped_image = Image.open(imgPath).crop((x_min, y_min, x_max, y_max))

    img_blurred = cv2.GaussianBlur(np.array(cropped_image), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blurred, threshold1=85, threshold2=120)

    edges_center = edges.T[int(len(edges.T) / 2)]

    if 255 in edges_center:
        attributes['Glasses'] = 'Present'
    else:
        attributes['Glasses'] = 'Absent'

    # Beard Detection
    face_cascade = cv2.CascadeClassifier(r"Pre-trained Models\haarcascade_frontalface_default.xml")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_image, 1.1, 5)

    for (x, y, w, h) in detected_faces:
        mask = np.zeros_like(image)
        mask = cv2.ellipse(mask, (int((x + w) / 1.2), y + h), (69, 69), 0, 0, -180, (255, 255, 255), thickness=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    beard_region = np.bitwise_and(image, mask)
    hsv_image = cv2.cvtColor(beard_region, cv2.COLOR_BGR2HSV)
    lower_black = np.array([94, 80, 2])
    upper_black = np.array([126, 255, 255])
    beard_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    if cv2.countNonZero(beard_mask) == 0:
        attributes['Facial Hair'] = 'Absent'
    else:
        attributes['Facial Hair'] = 'Present'

    # Iris Color Detection
    flag = 0

    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detected_faces_dlib = face_detector(gray_image, 0)

    for face in detected_faces_dlib:
        eyes = []

        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        shape = landmark_predictor(gray_image, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]

        eyes.append(left_eye)
        eyes.append(right_eye)

        for index, eye in enumerate(eyes):
            flag += 1
            left_side_eye = eye[0]
            right_side_eye = eye[3]
            top_side_eye = eye[1]
            bottom_side_eye = eye[4]

            eye_width = right_side_eye[0] - left_side_eye[0]
            eye_height = bottom_side_eye[1] - top_side_eye[1]

            eye_x1 = int(left_side_eye[0] - 0 * eye_width)
            eye_x2 = int(right_side_eye[0] + 0 * eye_width)

            eye_y1 = int(top_side_eye[1] - 1 * eye_height)
            eye_y2 = int(bottom_side_eye[1] + 0.75 * eye_height)

            roi_eye = image
            if flag == 1:
                break

    row, col, _ = roi_eye.shape

    iris_color_sample = roi_eye[row // 2:(row // 2) + 1, int((col // 3) + 3):int((col // 3)) + 6]
    iris_color_sample = iris_color_sample[0][2]
    iris_color_rgb = tuple(iris_color_sample)

    def get_color_name_from_rgb(rgb_color):
        min_colors = {}
        for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
            rd = (r_c - rgb_color[0]) ** 2
            gd = (g_c - rgb_color[1]) ** 2
            bd = (b_c - rgb_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

    iris_color_hex = '#{:02X}{:02X}{:02X}'.format(iris_color_rgb[0], iris_color_rgb[1], iris_color_rgb[2])
    attributes["Iris Color"] = iris_color_hex

    # Hair Color Detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    x, y, w, h = detected_faces[0]

    crop_x1 = max(x + 20, 0)
    crop_y1 = max(y - 50, 0)
    crop_x2 = min(x + w - 20, image.shape[1])
    crop_y2 = min(y + h, image.shape[0])

    cropped_hair_image = Image.fromarray(image[crop_y1:crop_y2, crop_x1:crop_x2])
    cropped_hair_image.save("temp_hair.png")

    color_thief = ColorThief('temp_hair.png')
    dominant_hair_color = color_thief.get_color(quality=1)

    hair_color_hex = '#{:02X}{:02X}{:02X}'.format(dominant_hair_color[0], dominant_hair_color[1], dominant_hair_color[2])
    attributes['Hair Color'] = hair_color_hex

    os.remove(imgPath)
    os.remove('temp_hair.png')

    return jsonify(attributes), 201

if __name__ == "__main__":
    app.run(debug=True)
