import cv2
import dlib
import numpy as np
import os
import sys
from screeninfo import get_monitors

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models")
video_dir = os.path.join(script_dir, "video")

# Get screen dimensions
try:
    screen = get_monitors()[0]
    SCREEN_WIDTH = screen.width
    SCREEN_HEIGHT = screen.height
except:
    print("Could not get screen info, using default values")
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

def resize_frame(frame, max_width, max_height):
    """Resize frame to fit within max dimensions while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    # Calculate aspect ratios
    aspect = width / height
    target_aspect = max_width / max_height
    
    # Calculate new dimensions
    if aspect > target_aspect:
        # Width limited by max_width
        new_width = max_width
        new_height = int(new_width / aspect)
    else:
        # Height limited by max_height
        new_height = max_height
        new_width = int(new_height * aspect)
    
    return cv2.resize(frame, (new_width, new_height)), new_width / width

def get_video_files():
    """Get all video files from the video directory"""
    if not os.path.exists(video_dir):
        print(f"Creating video directory at: {video_dir}")
        os.makedirs(video_dir)
        return []
    
    # List of common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    # Get all video files
    video_files = []
    for file in os.listdir(video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(video_dir, file))
    
    return video_files

def select_video(video_files):
    """Let user select a video file from the list"""
    if not video_files:
        print("No video files found in the video directory!")
        print(f"Please place video files in: {video_dir}")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv")
        sys.exit(1)
    
    if len(video_files) == 1:
        print(f"Found one video: {os.path.basename(video_files[0])}")
        return video_files[0]
    
    print("\nAvailable videos:")
    for i, video in enumerate(video_files, 1):
        print(f"{i}. {os.path.basename(video)}")
    
    while True:
        try:
            choice = int(input("\nSelect a video (enter number): "))
            if 1 <= choice <= len(video_files):
                return video_files[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def recognize_face(face_embedding, threshold=0.6):
    """Compare face embedding with known faces in database"""
    if not face_database:
        return None
    
    for face_id, stored_embedding in face_database.items():
        distance = np.linalg.norm(np.array(face_embedding) - np.array(stored_embedding))
        if distance < threshold:
            return face_id
    return None

def register_face(face_embedding):
    """Add new face to database"""
    global face_id_counter
    face_id = face_id_counter
    face_database[face_id] = face_embedding
    face_id_counter += 1
    return face_id

try:
    # Get and select video file
    video_files = get_video_files()
    video_path = select_video(video_files)
    print(f"\nSelected video: {os.path.basename(video_path)}")

    print("Initializing face detector...")
    detector = dlib.get_frontal_face_detector()
    
    # Check if model files exist and print their paths
    model_files = {
        "shape_predictor": os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat"),
        "face_recognition": os.path.join(model_dir, "dlib_face_recognition_resnet_model_v1.dat")
    }
    
    for name, file_path in model_files.items():
        print(f"Loading {name} from: {file_path}")
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            print("Please run setup.bat to download required models")
            sys.exit(1)
    
    print("Loading facial landmark predictor...")
    predictor = dlib.shape_predictor(model_files["shape_predictor"])
    
    print("Loading face recognition model...")
    recognizer = dlib.face_recognition_model_v1(model_files["face_recognition"])
    
    face_database = {}
    face_id_counter = 0

    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        sys.exit(1)

    # Calculate maximum dimensions (80% of screen size)
    MAX_WIDTH = int(SCREEN_WIDTH * 0.8)
    MAX_HEIGHT = int(SCREEN_HEIGHT * 0.8)

    # Create named window
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

    frame_count = 0
    print("Starting video processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        
        # Convert to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces on original frame
        faces = detector(rgb_frame, 1)
        
        # Resize frame for display
        display_frame, scale_factor = resize_frame(frame, MAX_WIDTH, MAX_HEIGHT)
        
        if len(faces) > 0:
            print(f"Frame {frame_count}: Detected {len(faces)} faces")

        for face in faces:
            try:
                # Get face landmarks
                shape = predictor(rgb_frame, face)
                face_embedding = recognizer.compute_face_descriptor(rgb_frame, shape)
                
                face_id = recognize_face(face_embedding)
                if face_id is None:
                    face_id = register_face(face_embedding)
                
                # Scale coordinates for display
                x = int(face.left() * scale_factor)
                y = int(face.top() * scale_factor)
                w = int(face.width() * scale_factor)
                h = int(face.height() * scale_factor)
                
                # Draw rectangle and ID on display frame
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {face_id}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing face in frame {frame_count}: {str(e)}")

        # Add frame counter to the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", display_frame)
        
        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)
