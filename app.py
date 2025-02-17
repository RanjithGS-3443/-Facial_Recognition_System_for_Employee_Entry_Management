import os
import cv2
import face_recognition
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime
import pandas as pd
import json
import logging
import argparse
import time

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('face_attendance/known_faces', exist_ok=True)
os.makedirs('face_attendance/attendance_records', exist_ok=True)

# Global variables to store face encodings
known_faces = []
known_names = []

# Load existing face encodings if available
if os.path.exists("face_encodings.pkl"):
    with open("face_encodings.pkl", "rb") as f:
        known_faces, known_names, known_ages, known_professions = pickle.load(f)
else:
    known_faces, known_names, known_ages, known_professions = [], [], [], []

# Add at the top with other global variables
recognition_log = []

class FaceAttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.student_details = {}  # Store additional student details
        self.attendance_df = None
        self.initialize_system()

    def initialize_system(self):
        # Create necessary directories if they don't exist
        if not os.path.exists('face_attendance/known_faces'):
            os.makedirs('face_attendance/known_faces')
        if not os.path.exists('face_attendance/attendance_records'):
            os.makedirs('face_attendance/attendance_records')
            
        # Initialize attendance DataFrame
        self.load_attendance_record()
        
        # Load known faces
        self.load_known_faces()

    def load_known_faces(self):
        """Load all known faces from the known_faces directory"""
        global known_faces, known_names
        known_faces = []
        known_names = []
        
        known_faces_dir = 'face_attendance/known_faces'
        print(f"Scanning directory: {known_faces_dir}")
        files = os.listdir(known_faces_dir)
        print(f"Found files: {files}")
        
        for filename in files:
            if filename.endswith('.json'):
                usn = filename.replace('.json', '')
                image_files = [f for f in os.listdir(known_faces_dir) if f.startswith(usn) and f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"Found image files for {usn}: {image_files}")
                
                if image_files:
                    image_path = os.path.join(known_faces_dir, image_files[0])
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            known_faces.append(face_encodings[0])
                            known_names.append(usn)
                            print(f"Successfully loaded face encoding for USN: {usn}")
                        else:
                            print(f"No face encodings found in image for USN: {usn}")
                    except Exception as e:
                        print(f"Error loading face for {usn}: {str(e)}")

    def load_attendance_record(self):
        """Load or create attendance record"""
        today = datetime.now().strftime('%Y-%m-%d')
        file_path = f'face_attendance/attendance_records/attendance_{today}.csv'
        
        if os.path.exists(file_path):
            self.attendance_df = pd.read_csv(file_path)
        else:
            self.attendance_df = pd.DataFrame(columns=[
                'empid', 'name', 'Position', 'Year_of_Joing', 
                'mobile_number_and_email', 'time', 'date'
            ])

    def mark_attendance(self, empid):
        """Mark attendance for a recognized person"""
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')
        current_date = now.strftime('%Y-%m-%d')
        
        # Get student details from stored data
        student_file = f"face_attendance/known_faces/{empid}.json"
        try:
            with open(student_file, 'r') as f:
                student_details = json.load(f)
            
            # Load or create attendance DataFrame
            file_path = f'face_attendance/attendance_records/attendance_{current_date}.csv'
            if os.path.exists(file_path):
                self.attendance_df = pd.read_csv(file_path)
            else:
                self.attendance_df = pd.DataFrame(columns=[
                    'empid', 'name', 'Position', 'Year_of_Joing', 
                'mobile_number_and_emaill', 'time', 'date'
                ])
            
            # Check if attendance already marked for today
            today_attendance = self.attendance_df[
                (self.attendance_df['empid'] == student_details['empid']) & 
                (self.attendance_df['date'] == current_date)
            ]
            
            if today_attendance.empty:
                new_attendance = pd.DataFrame({
                    'empid': [student_details['empid']],
                    'name': [student_details['name']],
                    'Position': [student_details['Position']],
                    'Year_of_Joing': [student_details['Year_of_Joing']],
                    'mobile_number_and_email': [student_details['mobile_number_and_email']],
                    'time': [current_time],
                    'date': [current_date]
                })
                
                self.attendance_df = pd.concat(
                    [self.attendance_df, new_attendance], 
                    ignore_index=True
                )
                
                # Save attendance record
                self.attendance_df.to_csv(file_path, index=False)
                print(f"Attendance marked for {student_details['name']}")
                return True, student_details['name']
            else:
                print(f"Attendance already marked for {student_details['name']} today")
                return False, student_details['name']
        except Exception as e:
            print(f"Error marking attendance: {str(e)}")
            return False, None

    def add_new_face(self, image_path, name, empid, Position, Year_of_Joing, mobile_number_and_email):
        """Add a new face to the known faces"""
        try:
            # Load and encode the face
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                print("No face detected in the image. Please try another image.")
                return False

            face_encoding = face_encodings[0]
            
            # Save the image to known_faces directory with USN as filename
            extension = os.path.splitext(image_path)[1]
            new_image_path = f'face_attendance/known_faces/{empid}{extension}'
            
            # Copy the image instead of moving it
            import shutil
            shutil.copy2(image_path, new_image_path)
            
            # Add to known faces
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(empid)  # Use USN as identifier
            
            # Store student details
            student_details = {
                "name": name,
                "empid": empid,
                "Position": Position,
                "Year_of_Joing": Year_of_Joing,
                "mobile_number_and_email": mobile_number_and_email
            }
            
            # Save student details to JSON file
            details_file = f'face_attendance/known_faces/{empid}.json'
            with open(details_file, 'w') as f:
                json.dump(student_details, f)
            
            print(f"Successfully added {name} to known faces")
            return True
        except Exception as e:
            print(f"Error adding new face: {str(e)}")
            return False

    def start_recognition(self):
        """Start the face recognition system using webcam"""
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Find all faces in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Process each face found in the frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding
                )
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    self.mark_attendance(name)

                # Draw rectangle around face and display name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Display the resulting frame
            cv2.imshow('Face Recognition Attendance System', frame)

            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

@app.route("/")
def index():
    app.logger.debug("Loading index page")
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    label = request.form["label"]
    age = request.form["age"]
    profession = request.form["profession"]

    if file and label:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        # Process the uploaded image
        image = face_recognition.load_image_file(filename)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            known_faces.append(face_encodings[0])
            known_names.append(label)
            known_ages.append(age)
            known_professions.append(profession)

            # Save updated encodings
            with open("face_encodings.pkl", "wb") as f:
                pickle.dump((known_faces, known_names, known_ages, known_professions), f)

            return "Image uploaded and face labeled successfully!"
        else:
            return "No face detected, please upload another image."

    return "Upload failed."

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["test_image"]
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        image = face_recognition.load_image_file(filename)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encodings[0])
            name = "Unknown"
            age = "N/A"
            profession = "N/A"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                age = known_ages[match_index]
                profession = known_professions[match_index]

            # Return the recognized details as a JSON response
            return jsonify({
                "name": name,
                "age": age,
                "profession": profession
            })

        else:
            return jsonify({"message": "No face detected."})

    return jsonify({"message": "Upload failed."})

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    process_frame = True
    last_recognition_time = 0
    recognition_cooldown = 3
    attendance_message = ""
    message_display_time = 0
    message_duration = 3  # seconds to display message

    # Add messages to the log
    def log_message(text, type='info'):
        global recognition_log
        recognition_log.append({
            'text': text,
            'type': type,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        # Keep only last 50 messages
        if len(recognition_log) > 50:
            recognition_log.pop(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        
        # Clear old messages
        if attendance_message and (current_time - message_display_time) > message_duration:
            attendance_message = ""

        if process_frame and (current_time - last_recognition_time) >= recognition_cooldown:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    student_usn = known_names[match_index]
                    
                    student_file = f"face_attendance/known_faces/{student_usn}.json"
                    if os.path.exists(student_file):
                        with open(student_file, 'r') as f:
                            student_details = json.load(f)
                            name = student_details['name']
                            
                            # Only try to mark attendance if enough time has passed
                            if (current_time - last_recognition_time) >= recognition_cooldown:
                                attendance_system = FaceAttendanceSystem()
                                success, student_name = attendance_system.mark_attendance(student_usn)
                                
                                if success:
                                    attendance_message = f"Attendance Marked: {student_name}"
                                    message_display_time = current_time
                                    last_recognition_time = current_time
                                    log_message(f"Successfully recognized: {student_name}", 'success')
                                else:
                                    if student_name:
                                        attendance_message = f"Already Marked: {student_name}"
                                        message_display_time = current_time
                                        last_recognition_time = current_time
                                        log_message(f"Attendance already marked for {student_name}", 'warning')

                # Scale back to original frame size
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display attendance message if exists
        if attendance_message:
            # Create semi-transparent overlay for message
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (600, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, attendance_message, 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)

        process_frame = not process_frame

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/camera")
def camera():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/register_face", methods=["POST"])
def register_face():
    try:
        file = request.files["image"]
        name = request.form["name"]
        empid = request.form["empid"]
        Position = request.form["Position"]
        Year_of_Joing = request.form["Year_of_Joing"]
        mobile_number_and_email = request.form["mobile_number_and_email"]

        if file and name:
            # Save the image file with USN as prefix
            extension = os.path.splitext(file.filename)[1]
            image_filename = f"{empid}{extension}"
            image_path = os.path.join('face_attendance/known_faces', image_filename)
            file.save(image_path)

            # Process the uploaded image
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                # Store student details
                student_details = {
                    "name": name,
                    "empid": empid,
                    "Position": Position,
                    "Year_of_Joing": Year_of_Joing,
                    "mobile_number_and_email": mobile_number_and_email
                }

                # Save student details to JSON
                with open(f"face_attendance/known_faces/{empid}.json", "w") as f:
                    json.dump(student_details, f)

                # Update known faces in memory
                known_faces.append(face_encodings[0])
                known_names.append(empid)

                return jsonify({"success": True, "message": "Registration successful!"})
            else:
                # Remove the saved image if no face detected
                os.remove(image_path)
                return jsonify({
                    "success": False, 
                    "message": "No face detected in the image"
                })

    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

@app.route("/get_attendance")
def get_attendance():
    try:
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        Position = request.args.get('Position', '')
        Year_of_Joing = request.args.get('Year_of_Joing', '')

        # Create the file path
        file_path = f'face_attendance/attendance_records/attendance_{date}.csv'

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"No attendance file found for date: {date}")
            return jsonify([])

        try:
            # Read attendance file
            attendance_df = pd.read_csv(file_path)
            
            # Convert column names to lowercase for consistency
            attendance_df.columns = attendance_df.columns.str.lower()
            
            # Apply filters if provided
            if Position:
                attendance_df = attendance_df[attendance_df['Position'].str.lower() == Position.lower()]
            if Year_of_Joing:
                attendance_df = attendance_df[attendance_df['Year_of_Joing'] == int(Year_of_Joing)]

            # Convert to records
            records = attendance_df.to_dict('records')
            print(f"Found {len(records)} attendance records for date: {date}")
            return jsonify(records)

        except pd.errors.EmptyDataError:
            print(f"Empty attendance file for date: {date}")
            return jsonify([])

    except Exception as e:
        print(f"Error getting attendance: {str(e)}")
        return jsonify([])

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})
    
    try:
        print("Processing attendance request...")
        file = request.files['image']
        npimg = np.fromfile(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        print("Converting image to RGB...")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print("Detecting faces...")
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            print("No faces detected in image")
            return jsonify({'success': False, 'error': 'No face detected'})
            
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        print(f"Found {len(face_encodings)} face(s) in image")
        
        print(f"Number of known faces: {len(known_faces)}")
        print(f"Known names: {known_names}")
        
        # Compare with known faces
        for face_encoding in face_encodings:
            if not known_faces:
                print("No registered faces found in database")
                return jsonify({'success': False, 'error': 'No registered faces found'})
                
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            print(f"Face matches: {matches}")
            
            if True in matches:
                match_index = matches.index(True)
                student_usn = known_names[match_index]
                print(f"Matched with student USN: {student_usn}")
                
                attendance_system = FaceAttendanceSystem()
                success, student_name = attendance_system.mark_attendance(student_usn)
                print(f"Attendance marking result: success={success}, name={student_name}")
                
                if success:
                    return jsonify({
                        'success': True,
                        'student_name': student_name,
                        'message': 'Attendance marked successfully!'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'student_name': student_name,
                        'error': 'Attendance already marked for today'
                    })
            else:
                print("No matching face found in database")
            
        return jsonify({
            'success': False,
            'error': 'Face not recognized'
        })
        
    except Exception as e:
        print(f"Error processing attendance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_recognition_log')
def get_recognition_log():
    return jsonify({'messages': recognition_log})

def main():
    parser = argparse.ArgumentParser(description='Face Attendance System')
    parser.add_argument('--cli', action='store_true', help='Run in command-line mode')
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_web()

def run_cli():
    attendance_system = FaceAttendanceSystem()
    # Original CLI code here...

def run_web():
    app.logger.info("Starting Face Attendance System Web Interface")
    os.makedirs('face_attendance/known_faces', exist_ok=True)
    os.makedirs('face_attendance/attendance_records', exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Create an instance of FaceAttendanceSystem to load known faces
    attendance_system = FaceAttendanceSystem()
    attendance_system.load_known_faces()
    
    # Verify that faces were loaded
    print(f"Loaded {len(known_faces)} face(s)")
    print(f"Known names: {known_names}")
    
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()
