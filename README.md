# PersonaScan API 
PersonaScan is a powerful API designed to analyze and extract various attributes from images, including gender, race, emotion, age, glasses presence, facial hair, iris color, and hair color. It leverages various state-of-the-art computer vision and deep learning techniques such as Dlib, OpenCV, Deepface Library and some other pretrained models to provide detailed insights into facial features. 
## Features 
- **Gender Detection**: Identifies the dominant gender in the image. 
 - **Race Detection**: Analyzes the person's race in the image. 
 - **Emotion Detection**: Detects the person's emotion in the image. 
 - **Age Estimation**: Estimates the age of the individual in the image. 
 - **Glasses Detection**: Determines if the person is wearing glasses in the image. 
 - **Facial Hair Detection**: Identifies the presence of facial hair in the image. 
 - **Iris Color Detection**: Determines the color of the iris in the image. 
 - **Hair Color Detection**: Analyzes the color of the hair in the image. *(Still in development)*
## Getting Started 
To use the PersonaScan API, follow these steps: 
### Prerequisites 
- Python 3.x 
 - Flask 
 - OpenCV 
 - dlib 
 - DeepFace 
- PIL 
- imutils 
 - colorthief 
- webcolors 
### Installation 
1. Clone the repository: 
```bash
git clone https://github.com/yourusername/personascan-api.git
```
```bash
 cd personascan-api
```
3. Install the required packages
```bash
 pip install -r requirements.txt
```
Note: Install the Dlib library for your python version from [here](!https://github.com/z-mahmud22/Dlib_Windows_Python3.x).
### Running the API
1. Start the flask server.
``` bash
python app.py
```
The server will run on `http://127.0.0.1:5000`.
### API Endpoints
#### `POST /analyze`

Analyzes the attributes of the given image.
It takes the input from the request body which is a base64 string of the image. The response is json containing the description of various facial features.
#### Request Body:
```json 
{ "b64_string": ".....base64....string...of...the....image......" }
```
#### Sample Response:
```json 
{ "Gender": "Male", "Race": "Caucasian", "Emotion": "Happy", "Age": 29, "Glasses": "Present", "Facial Hair": "Absent", "Iris Color": "#7F3F6D", "Hair Color": "#C9A29D" }
```
#### Status Codes:
- ```201 Created``` - If the analysis is successful.
- ```400 Bad Request``` - If the request format is incorrect or required fields are missing.
