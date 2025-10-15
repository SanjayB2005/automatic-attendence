import os
import cv2
import face_recognition
import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO

class StudentEnrollment:
    def __init__(self, embeddings_file='student_embeddings.pkl', models_dir='uploads/students'):
        self.embeddings_file = embeddings_file
        self.models_dir = models_dir
        self.known_face_encodings = []
        self.known_student_ids = []
        self.load_embeddings()
    
    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_student_ids = data['student_ids']
    
    def save_embeddings(self):
        data = {
            'encodings': self.known_face_encodings,
            'student_ids': self.known_student_ids
        }
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def process_enrollment_photo(self, student_id):
        """Process a new student's enrollment photo from models directory and add to embeddings"""
        try:
            # Construct path to photo in models directory
            photo_path = os.path.join(self.models_dir, f'{student_id}.jpg')
            if not os.path.exists(photo_path):
                return {
                    'success': False,
                    'message': f'Photo not found for student ID: {student_id}'
                }
                
            # Load and process image
            image = face_recognition.load_image_file(photo_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return {
                    'success': False,
                    'message': 'No face detected in the photo'
                }
            
            if len(face_locations) > 1:
                return {
                    'success': False,
                    'message': 'Multiple faces detected. Please upload a photo with only one face'
                }
            
            # Get face encoding
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            # Check for existing face matches
            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=0.6
                )
                
                if True in matches:
                    matched_id = self.known_student_ids[matches.index(True)]
                    return {
                        'success': False,
                        'message': f'Face already registered with student ID: {matched_id}'
                    }
            
            # Add new face encoding
            self.known_face_encodings.append(face_encoding)
            self.known_student_ids.append(student_id)
            
            # Save updated embeddings
            self.save_embeddings()
            
            # Process and save cropped face
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            
            # Save processed face image
            output_dir = 'processed_faces'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{student_id}.jpg')
            cv2.imwrite(output_path, face_image)
            
            return {
                'success': True,
                'message': 'Face enrolled successfully',
                'face_image_path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing photo: {str(e)}'
            }

if __name__ == "__main__":
    # Test the enrollment process
    enrollment = StudentEnrollment()
    
    test_student_id = input("Enter student ID: ")
    
    result = enrollment.process_enrollment_photo(test_student_id)
    print(result)