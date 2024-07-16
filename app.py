import cv2
from deepface import DeepFace
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import tempfile
import os
import threading

app = Flask(__name__)

progress = 0

def generate_frames(video_source, output_filename):
    global progress
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    frame_number = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_interval = int(fps / 3)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    last_bounding_boxes = []
    last_emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            for (x, y, x2, y2), emotion in zip(last_bounding_boxes, last_emotions):
                font_scale = 9 * (width / 1080)
                thickness = int(30 * (width / 1080))
                text = f'Emotion: {emotion}'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

            out.write(frame)
            continue

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            last_bounding_boxes = []
            last_emotions = []

            for face_info in analysis:
                face = face_info['region']
                emotion = face_info['dominant_emotion']

                x, y, width, height = face['x'], face['y'], face['w'], face['h']
                x2, y2 = x + width, y + height

                last_bounding_boxes.append((x, y, x2, y2))
                last_emotions.append(emotion)

                font_scale = int(0.6 * (width / 350))
                thickness = int(2 * (width / 350))

                text = f'Emotion: {emotion}'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

        except Exception as e:
            print(f"Error processing frame: {e}")

        out.write(frame)

        progress = int((frame_number / total_frames) * 100)

    cap.release()
    out.release()
    progress = 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global progress
    progress = 0
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        output_filename = 'output_' + file.filename

        thread = threading.Thread(target=generate_frames, args=(temp_path, output_filename))
        thread.start()

        return jsonify({'message': 'Processing started', 'video_path': output_filename})

@app.route('/progress')
def get_progress():
    global progress
    return jsonify({'progress': progress})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(tempfile.gettempdir(), filename, as_attachment=True)

def generate_webcam_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            for face_info in analysis:
                face = face_info['region']
                emotion = face_info['dominant_emotion']

                x, y, width, height = face['x'], face['y'], face['w'], face['h']
                x2, y2 = x + width, y + height

                # Adjust font scale and thickness based on the frame width
                frame_width = frame.shape[1]
                font_scale = 0.6 * (frame_width / 350)
                thickness = int(2 * (frame_width / 350))

                text = f'Emotion: {emotion}'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

        except Exception as e:
            print(f"Error processing frame: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
