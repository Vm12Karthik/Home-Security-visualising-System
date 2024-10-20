import cv2
import time
import datetime
import sounddevice as sd
import numpy as np
import wavio

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5
SAMPLE_RATE = 44100  # Sample rate for audio

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

previous_frame = None
out = None
audio_out = None

def record_audio(filename, duration):
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, audio_data, SAMPLE_RATE, sampwidth=2)  # Save as WAV file

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if previous_frame is None:
        previous_frame = gray
        continue
    diff_frame = cv2.absdiff(previous_frame, gray)
    previous_frame = gray
    _, thresh = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) + len(bodies) > 0 or len(contours) > 0:
        if not detection:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            audio_out = f"{current_time}.wav"
            print("Started Recording Video and Audio!")
            # Start recording audio for the duration
            record_audio(audio_out, SECONDS_TO_RECORD_AFTER_DETECTION)
        timer_started = False  
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                if out:  
                    out.release()
                    out = None
                print('Stop Recording Video!')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection and out:
        out.write(frame)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

if out:  
    out.release()
cap.release()
cv2.destroyAllWindows()