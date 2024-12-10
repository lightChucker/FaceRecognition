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

# Global variables
face_labels = {}  # Dictionary to store face ID -> name mappings
paused = False    # Global pause state

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
    """
    Resize frame while maintaining aspect ratio
    Returns resized frame and scale factor
    """
    height, width = frame.shape[:2]
    
    # Calculate aspect ratios
    aspect_ratio = width / height
    target_ratio = max_width / max_height
    
    # Determine new dimensions while maintaining aspect ratio
    if aspect_ratio > target_ratio:
        # Width is the limiting factor
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:
        # Height is the limiting factor
        new_height = max_height
        new_width = int(max_height * aspect_ratio)
    
    # Calculate scale factor for face detection coordinates
    scale_factor = new_width / width
    
    # Resize frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_frame, scale_factor

def get_available_cameras():
    """Get a list of available webcam devices"""
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        ret, _ = cap.read()
        if ret:
            camera_name = f"Camera {index}"
            available_cameras.append((index, camera_name))
        cap.release()
        index += 1
    return available_cameras

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

def select_input_source():
    """Let user choose between webcam or video file"""
    print("\nSelect input source:")
    print("1. Webcam")
    print("2. Video file")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1 or 2): "))
            if choice in [1, 2]:
                break
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    if choice == 1:
        cameras = get_available_cameras()
        if not cameras:
            print("No webcams detected!")
            return None, None
            
        print("\nAvailable cameras:")
        for idx, name in cameras:
            print(f"{idx + 1}. {name}")
            
        while True:
            try:
                cam_choice = int(input("\nSelect a camera (enter number): ")) - 1
                if 0 <= cam_choice < len(cameras):
                    return "webcam", cameras[cam_choice][0]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    else:
        video_files = get_video_files()
        selected_video = select_video(video_files)
        return "video", selected_video

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

def relabel_face():
    """Allow user to relabel face IDs"""
    print("\nCurrent face labels:")
    for face_id in face_database.keys():
        label = face_labels.get(face_id, f"ID {face_id}")
        print(f"ID {face_id}: {label}")
    
    while True:
        try:
            face_id = input("\nEnter face ID to relabel (or 'q' to quit): ")
            if face_id.lower() == 'q':
                break
            face_id = int(face_id)
            if face_id in face_database:
                new_label = input(f"Enter new label for ID {face_id}: ").strip()
                if new_label:
                    face_labels[face_id] = new_label
                    print(f"ID {face_id} relabeled as: {new_label}")
            else:
                print("Face ID not found!")
        except ValueError:
            print("Please enter a valid number or 'q'")

def draw_face_info(frame, x, y, w, h, face_id):
    """Draw face rectangle and label"""
    label = face_labels.get(face_id, f"ID: {face_id}")
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw background rectangle for text
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (x, y-text_height-10), (x+text_width+10, y), (0, 255, 0), -1)
    
    # Draw text
    cv2.putText(frame, label, (x+5, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

try:
    # Get input source
    source_type, source = select_input_source()
    if source is None:
        print("No valid input source available.")
        sys.exit(1)

    print("\nInitializing face detector...")
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

    print(f"Opening {'webcam' if source_type == 'webcam' else 'video file'}...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open input source")
        sys.exit(1)

    # Calculate maximum dimensions (80% of screen size)
    MAX_WIDTH = int(SCREEN_WIDTH * 0.8)
    MAX_HEIGHT = int(SCREEN_HEIGHT * 0.8)

    # Create named window
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

    frame_count = 0
    is_webcam = source_type == "webcam"

    print("\nControls:")
    print("'p' - Pause/Unpause")
    print("'r' - Relabel faces (while paused)")
    print("'q' - Quit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    print("Error reading from webcam!")
                    break
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
                    
                    # Draw face info
                    draw_face_info(display_frame, x, y, w, h, face_id)
                    
                except Exception as e:
                    print(f"Error processing face in frame {frame_count}: {str(e)}")

            # Add frame counter to the video
            if not is_webcam:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add pause indicator
        if paused:
            cv2.putText(display_frame, "PAUSED", (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("\nProgram paused")
                print("Press 'p' to unpause")
                print("Press 'r' to relabel faces")
                print("Press 'q' to quit")
        elif key == ord('r') and paused:
            relabel_face()

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)
