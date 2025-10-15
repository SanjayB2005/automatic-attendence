from flask import Flask, request, jsonify, Response, send_file
import os
import cv2
import numpy as np
from PIL import Image
import io
import pickle
from datetime import datetime
from flask_cors import CORS
import face_recognition

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
import os
import cv2
import numpy as np
from PIL import Image
import io
import pickle
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configure storage
UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_FILE = 'student_embeddings.pkl'
PROCESSED_FACES = 'processed_faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FACES, exist_ok=True)

def load_student_data():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {
        'encodings': [],
        'student_ids': []
    }

def save_student_data(data):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(data, f)

# Load student data
student_data = load_student_data()

def get_face_encoding(image):
    """Get face encoding using face_recognition library - returns largest face only"""
    # Convert BGR to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find face locations
    face_locations = face_recognition.face_locations(rgb_image)
    
    if len(face_locations) == 0:
        print(f"No faces detected in image of size: {image.shape}")
        return None
    
    print(f"Detected {len(face_locations)} faces")
    
    # Get the largest face (by area)
    largest_face_idx = 0
    largest_area = 0
    for i, (top, right, bottom, left) in enumerate(face_locations):
        area = (bottom - top) * (right - left)
        if area > largest_area:
            largest_area = area
            largest_face_idx = i
    
    top, right, bottom, left = face_locations[largest_face_idx]
    print(f"Largest face: top={top}, right={right}, bottom={bottom}, left={left}")
    
    # Get face encoding
    face_encodings = face_recognition.face_encodings(rgb_image, [face_locations[largest_face_idx]])
    
    if len(face_encodings) > 0:
        # Convert face_recognition coordinates to OpenCV format for consistency
        x, y, w, h = left, top, right - left, bottom - top
        return face_encodings[0], (x, y, w, h)
    
    return None

def get_all_face_encodings(image):
    """Get encodings for all faces in the image using face_recognition library"""
    # Convert BGR to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find face locations
    face_locations = face_recognition.face_locations(rgb_image)
    
    if len(face_locations) == 0:
        print(f"No faces detected in image of size: {image.shape}")
        return []
    
    print(f"Raw detection: {len(face_locations)} faces found")
    
    # Filter out faces that are too small or too large
    valid_face_locations = []
    image_area = image.shape[0] * image.shape[1]
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Check if face is reasonable size
        face_area = (bottom - top) * (right - left)
        face_ratio = face_area / image_area
        
        if face_ratio < 0.01 or face_ratio > 0.5:  # Face should be 1-50% of image
            print(f"Face {i+1} rejected: size ratio {face_ratio:.3f} (too small/large)")
            continue
        
        valid_face_locations.append((top, right, bottom, left))
        # Convert to OpenCV coordinates for logging
        x, y, w, h = left, top, right - left, bottom - top
        print(f"Face {i+1} accepted: x={x}, y={y}, w={w}, h={h}, ratio={face_ratio:.3f}")
    
    print(f"After filtering: {len(valid_face_locations)} valid faces")
    
    if len(valid_face_locations) == 0:
        return []
    
    # Get face encodings for all valid faces
    face_encodings_raw = face_recognition.face_encodings(rgb_image, valid_face_locations)
    
    # Format results
    face_encodings = []
    for i, (encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings_raw, valid_face_locations)):
        # Convert face_recognition coordinates to OpenCV format
        x, y, w, h = left, top, right - left, bottom - top
        print(f"Processing valid face {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        face_encodings.append({
            'encoding': encoding,
            'coordinates': (x, y, w, h)
        })
    
    return face_encodings

def compare_faces(known_encoding, face_encoding, tolerance=0.6):
    """Compare faces using face_recognition library's built-in comparison"""
    if known_encoding is None or face_encoding is None:
        return False
    
    # Check if encodings are valid (should be 128-dimensional)
    if len(known_encoding) != 128 or len(face_encoding) != 128:
        print(f"Invalid encoding dimensions: {len(known_encoding)} vs {len(face_encoding)}")
        return False
    
    # Use face_recognition's built-in comparison
    matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
    distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
    
    print(f"Face distance: {distance:.3f}, tolerance: {tolerance}, match: {matches[0]}")
    
    return matches[0]

def get_face_confidence(known_encoding, face_encoding):
    """Calculate confidence score based on face distance using face_recognition library"""
    if known_encoding is None or face_encoding is None:
        return 0.0
    
    if len(known_encoding) != 128 or len(face_encoding) != 128:
        return 0.0
    
    # Use face_recognition's distance calculation
    distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
    
    # Convert distance to confidence (lower distance = higher confidence)
    # face_recognition typically uses distances from 0.0 to 1.0+
    # 0.0 = perfect match, 0.6 = default threshold, 1.0+ = poor match
    if distance <= 0.4:
        confidence = 1.0 - (distance / 0.4)  # Scale 0.0-0.4 to 1.0-0.0
    elif distance <= 0.6:
        confidence = 0.6 - (distance - 0.4) * 3  # Scale 0.4-0.6 to 0.6-0.0
    else:
        confidence = max(0.0, 0.3 - (distance - 0.6) * 0.5)  # Scale 0.6+ to diminishing confidence
    
    return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1

# Initialize systems
from werkzeug.utils import secure_filename

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Try to open the camera
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        if camera_available:
            cap.release()
            
        return jsonify({
            'status': 'healthy',
            'camera_available': camera_available,
            'opencv_version': cv2.__version__
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'opencv_version': cv2.__version__
        }), 500

@app.route('/enroll', methods=['POST'])
def enroll_student():
    if 'photo' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No photo file provided'
        }), 400
    
    photo = request.files['photo']
    student_id = request.form.get('student_id')
    force_enroll = request.form.get('force_enroll', 'false').lower() == 'true'
    
    if not student_id:
        return jsonify({
            'success': False,
            'message': 'No student ID provided'
        }), 400
    
    if photo.filename == '':
        return jsonify({
            'success': False,
            'message': 'No photo selected'
        }), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(f"{student_id}_{photo.filename}")
        photo_path = os.path.join(UPLOAD_FOLDER, filename)
        photo.save(photo_path)
        
        # Read image and detect face
        img = cv2.imread(photo_path)
        if img is None:
            os.remove(photo_path)
            return jsonify({
                'success': False,
                'message': 'Could not read uploaded image'
            }), 400
        
        # Get face encoding
        result = get_face_encoding(img)
        if result is None:
            os.remove(photo_path)
            return jsonify({
                'success': False,
                'message': 'No face detected in photo'
            }), 400
        
        face_encoding, (x, y, w, h) = result
        
        # Check for duplicate faces (with more lenient tolerance)
        duplicate_found = False
        duplicate_student_id = None
        for i, encoding in enumerate(student_data['encodings']):
            if compare_faces(encoding, face_encoding, tolerance=0.6):  # Standard face_recognition tolerance
                duplicate_found = True
                duplicate_student_id = student_data["student_ids"][i]
                break
        
        if duplicate_found and not force_enroll:
            # Still save the processed face image for debugging
            face_img = img[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (200, 200))
            face_path = os.path.join(PROCESSED_FACES, f'{student_id}_duplicate.jpg')
            cv2.imwrite(face_path, face_img_resized)
            
            os.remove(photo_path)
            return jsonify({
                'success': False,
                'message': f'Face similar to student ID: {duplicate_student_id}. Use force_enroll=true to override this check.'
            }), 400
        
        # Check if student ID already exists and remove old entry
        if student_id in student_data['student_ids']:
            # Remove existing entry for this student ID
            old_index = student_data['student_ids'].index(student_id)
            student_data['encodings'].pop(old_index)
            student_data['student_ids'].pop(old_index)
            print(f"Removed existing enrollment for student {student_id}")
        
        # Save face encoding and student ID
        student_data['encodings'].append(face_encoding)
        student_data['student_ids'].append(student_id)
        save_student_data(student_data)
        
        # Save processed face image (resize to standard size for consistency)
        face_img = img[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (200, 200))  # Standard size
        face_path = os.path.join(PROCESSED_FACES, f'{student_id}.jpg')
        cv2.imwrite(face_path, face_img_resized)
        
        print(f"Processed face saved to: {face_path}")
        
        # Clean up uploaded file
        os.remove(photo_path)
        
        return jsonify({
            'success': True,
            'message': 'Face enrolled successfully'
        })
            
    except Exception as e:
        # Clean up on error
        if os.path.exists(photo_path):
            os.remove(photo_path)
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/attendance/live')
def video_feed():
    def generate_frames():
        print("Attempting to open camera...")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Failed to open camera!")
            return b''
        
        print("Camera opened successfully!")
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera")
                break
            
            # Use the improved face detection for live video
            face_encodings = get_all_face_encodings(frame)
            
            # Process each detected face
            for face_data in face_encodings:
                face_encoding = face_data['encoding']
                coordinates = face_data['coordinates']
                x, y, w, h = coordinates
                
                # Find the best match with confidence scoring
                best_match = None
                best_confidence = 0.0
                best_distance = float('inf')
                
                for i, encoding in enumerate(student_data['encodings']):
                    distance = face_recognition.face_distance([encoding], face_encoding)[0]
                    confidence = get_face_confidence(encoding, face_encoding)
                    
                    # Find the match with the smallest distance (best match)
                    if distance < best_distance and distance < 0.8:  # Only consider reasonable matches (face_recognition scale)
                        best_distance = distance
                        best_confidence = confidence
                        best_match = {
                            'student_id': student_data['student_ids'][i],
                            'confidence': confidence
                        }
                
                # Only show recognition if confidence is high enough
                if best_match and best_confidence >= 0.4:
                    # Draw green rectangle and student ID
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{best_match['student_id']} ({best_confidence:.2f})"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                else:
                    # Draw red rectangle for unrecognized faces
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Make sure to release the camera
        camera.release()
        print("Camera released")
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/enroll-from-camera', methods=['POST'])
def enroll_from_camera():
    """Enroll a student from the current camera frame"""
    if 'photo' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No photo file provided'
        }), 400
    
    photo = request.files['photo']
    student_id = request.form.get('student_id')
    
    if not student_id:
        return jsonify({
            'success': False,
            'message': 'No student ID provided'
        }), 400
    
    if photo.filename == '':
        return jsonify({
            'success': False,
            'message': 'No photo selected'
        }), 400
    
    try:
        # Read image file
        img = Image.open(photo.stream)
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        print(f"Enrolling student {student_id} from camera frame of size: {img_cv.shape}")
        
        # Get all face encodings
        face_encodings = get_all_face_encodings(img_cv)
        if not face_encodings:
            return jsonify({
                'success': False,
                'message': 'No faces detected in camera frame'
            }), 400
        
        # Use the largest face for enrollment
        largest_face = max(face_encodings, key=lambda f: f['coordinates'][2] * f['coordinates'][3])
        face_encoding = largest_face['encoding']
        coordinates = largest_face['coordinates']
        
        print(f"Using largest face at coordinates {coordinates} for enrollment")
        
        # Check for duplicate faces
        for i, encoding in enumerate(student_data['encodings']):
            if compare_faces(encoding, face_encoding):
                return jsonify({
                    'success': False,
                    'message': f'Face already registered with student ID: {student_data["student_ids"][i]}'
                }), 400
        
        # Save face encoding and student ID
        student_data['encodings'].append(face_encoding)
        student_data['student_ids'].append(student_id)
        save_student_data(student_data)
        
        print(f"Successfully enrolled student {student_id}")
        
        return jsonify({
            'success': True,
            'message': f'Student {student_id} enrolled successfully from camera frame',
            'faces_detected': len(face_encodings),
            'enrollment_face_coordinates': {
                'x': int(coordinates[0]),
                'y': int(coordinates[1]),
                'w': int(coordinates[2]),
                'h': int(coordinates[3])
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/test-face-detection', methods=['POST'])
def test_face_detection():
    """Test endpoint to check if faces can be detected in an image"""
    if 'photo' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No photo file provided'
        }), 400
    
    photo = request.files['photo']
    
    if photo.filename == '':
        return jsonify({
            'success': False,
            'message': 'No photo selected'
        }), 400
    
    try:
        # Read image file
        img = Image.open(photo.stream)
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        print(f"Testing face detection on image of size: {img_cv.shape}")
        
        # Get face encoding
        result = get_face_encoding(img_cv)
        if result is None:
            return jsonify({
                'success': False,
                'message': 'No face detected in photo',
                'image_size': img_cv.shape
            }), 400
        
        face_encoding, (x, y, w, h) = result
        
        return jsonify({
            'success': True,
            'message': 'Face detected successfully',
            'image_size': img_cv.shape,
            'face_coordinates': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'face_encoding_size': len(face_encoding)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/verify', methods=['POST'])
def verify_face():
    if 'photo' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No photo file provided'
        }), 400
    
    photo = request.files['photo']
    
    if photo.filename == '':
        return jsonify({
            'success': False,
            'message': 'No photo selected'
        }), 400
    
    try:
        # Read image file
        img = Image.open(photo.stream)
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        print(f"Processing image of size: {img_cv.shape}")
        print(f"Number of enrolled students: {len(student_data['encodings'])}")
        
        # Get all face encodings
        face_encodings = get_all_face_encodings(img_cv)
        if not face_encodings:
            return jsonify({
                'success': False,
                'message': 'No faces detected in photo'
            }), 400
        
        print(f"Processing {len(face_encodings)} faces for recognition")
        
        # Process ALL faces, not just the largest one
        all_recognized = []
        
        for face_idx, face_data in enumerate(face_encodings):
            face_encoding = face_data['encoding']
            coordinates = face_data['coordinates']
            
            print(f"\nProcessing face {face_idx + 1} at coordinates {coordinates}")
            
            # Find the best match for this face
            best_match = None
            best_distance = float('inf')
            
            for i, encoding in enumerate(student_data['encodings']):
                distance = face_recognition.face_distance([encoding], face_encoding)[0]
                confidence = get_face_confidence(encoding, face_encoding)
                
                print(f"  Student {student_data['student_ids'][i]}: distance={distance:.3f}, confidence={confidence:.3f}")
                
                # Find the match with the smallest distance (best match)
                if distance < best_distance and distance < 0.8:  # Only consider reasonable matches (face_recognition scale)
                    best_distance = distance
                    best_match = {
                        'student_id': student_data['student_ids'][i],
                        'confidence': confidence,
                        'distance': distance,
                        'face_coordinates': {
                            'x': int(coordinates[0]),
                            'y': int(coordinates[1]),
                            'w': int(coordinates[2]),
                            'h': int(coordinates[3])
                        }
                    }
                    print(f"    → New best match: {student_data['student_ids'][i]} (distance: {distance:.2f})")
            
            # Add this face's best match if it's good enough
            if best_match and best_match['confidence'] >= 0.4:
                all_recognized.append(best_match)
                print(f"✅ Face {face_idx + 1} recognized as: {best_match['student_id']} with confidence {best_match['confidence']:.3f}")
            else:
                print(f"❌ Face {face_idx + 1}: No good match found (best distance: {best_distance:.2f})")
        
        # Remove duplicates (same student recognized multiple times)
        unique_recognized = []
        seen_student_ids = set()
        for rec in all_recognized:
            if rec['student_id'] not in seen_student_ids:
                unique_recognized.append(rec)
                seen_student_ids.add(rec['student_id'])
        
        if unique_recognized:
            result = {
                'success': True,
                'recognized': unique_recognized,
                'total_faces_detected': len(face_encodings),
                'total_students_recognized': len(unique_recognized)
            }
        else:
            result = {
                'success': False,
                'message': f'No faces recognized in photo (detected {len(face_encodings)} faces)'
            }
        
        print(f"Verification result: {result}")
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)