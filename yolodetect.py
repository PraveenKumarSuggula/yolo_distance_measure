from flask import Flask, render_template, Response, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nikhilesh'
app.config['UPLOAD_FOLDER'] = 'static/files' 


class VideoUploadForm(FlaskForm):
    video_file = FileField("Video File", validators=[InputRequired()])
    submit = SubmitField("Upload")


def calculate_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width


def process_video_frames(video_path=None):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Attempt to load YOLOv4
    try:
        model_v4 = YOLO("../YOLO-Weights/yolov4.pt")
        model = model_v4
    except FileNotFoundError:
        # Load YOLOv8n if YOLOv4 is not found
        model_v8n = YOLO("../YOLO-Weights/yolov8n.pt")
        model = model_v8n

    # Assuming you know the width of an object in the real world (e.g., in meters)
    known_width = 1.0  # Replace with the actual known width of the object

    # Assuming you know the focal length of your camera (you might need to calibrate this)
    focal_length = 1000.0  # Replace with the actual focal length of your camera

    class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Choose the appropriate model based on confidence
        results = model(frame, stream=True)
        if hasattr(results, 'boxes'):
            boxes = results.boxes
        else:
            # Handle the case when results is a generator (e.g., YOLOv8n)
            boxes = [result.boxes for result in results]

        for box in boxes:
            for i in range(len(box.cls)):
                x1, y1, x2, y2 = map(int, box.xyxy[i])
                class_label = f'{class_names[int(box.cls[i])]} {math.ceil((box.conf[i] * 100)) / 100}'

                # Check confidence score
                if box.conf[i] < 0.85:
                    # Switch to YOLOv8n if confidence is below 0.85
                    results_v8n = model_v8n(frame, stream=True)
                    boxes_v8n = [result.boxes for result in results_v8n]
                    model_label = 'Model: yolov8n'
                else:
                    model_label = 'Model: yolov4'

                # Calculate distance (approximate)
                pixel_width = x2 - x1
                distance = calculate_distance(known_width, focal_length, pixel_width)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Draw label on the right side of the detection
                t_size = cv2.getTextSize(class_label, 0, fontScale=1, thickness=2)[0]
                c2 = x2 + 5, y1 + t_size[1] + 3
                cv2.rectangle(frame, (x2, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(frame, class_label, (x2 + 5, y1 + t_size[1]), 0, 1, [0, 0, 255], thickness=1,
                            lineType=cv2.LINE_AA)

                # Display distance on the frame
                distance_text = f'Distance: {round((distance / 2), 2)} meters'
                cv2.putText(frame, distance_text, (x2 + 5, y1 + t_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)

                # Display model type
                cv2.putText(frame, model_label, (x2 + 5, y1 + t_size[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)

        yield frame

    cap.release()
    cv2.destroyAllWindows()


def generate_video_frames(video_path=''):
    video_frames = process_video_frames(video_path)
    for frame in video_frames:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def index():
    session.clear()
    form = VideoUploadForm()
    if form.validate_on_submit():
        video_file = form.video_file.data
        video_file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename)))
        session['video_path'] = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
    return render_template('index.html', form=form)


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(video_path=session.get('video_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
