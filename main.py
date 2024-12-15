import cv2
import time
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import mediapipe as mp

class MultiPersonTracker:
    def __init__(self, video_path):
        # Initialize YOLO model for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True
        )
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        output_path = 'output_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.mp4'
        self.writer = cv2.VideoWriter(output_path, 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    self.fps, 
                                    (self.frame_width, self.frame_height))
        
        # Define person positions
        self.persons = {
            'TANVIR':    {'zone': (0.1, 0.3)},
            'SHAFAYET':  {'zone': (0.2, 0.4)},
            'TOUFIQ':    {'zone': (0.3, 0.5)},
            'FAISAL':    {'zone': (0.4, 0.6)},
            'MUFRAD':    {'zone': (0.5, 0.7)},
            'ANIK':      {'zone': (0.6, 0.8)},
            'IMRAN':     {'zone': (0.7, 0.9)},
            'EMON':      {'zone': (0.8, 1.0)}
        }
        
        # Initialize hand raise history
        self.hand_raise_history = deque(maxlen=5)
        
        print(f"Video loaded: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"Output will be saved to: {output_path}")

    def detect_people_and_poses(self, frame):
        """
        Detect people using YOLO and their poses using MediaPipe
        """
        # YOLO detection
        yolo_results = self.yolo(frame, classes=[0])  # class 0 is person
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        detected_people = []
        
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center position
                center_x = (x1 + x2) / (2 * self.frame_width)
                center_y = (y1 + y2) / (2 * self.frame_height)
                
                # Identify person based on position
                person_name = self.identify_person(center_x)
                
                detected_people.append({
                    'name': person_name,
                    'position': (center_x, center_y),
                    'box': (x1, y1, x2, y2),
                    'conf': float(box.conf),
                    'pose_landmarks': pose_results.pose_landmarks if pose_results.pose_landmarks else None
                })
        
        return detected_people

    def identify_person(self, x_position):
        """
        Identify person based on their x-position
        """
        for name, data in self.persons.items():
            if data['zone'][0] <= x_position <= data['zone'][1]:
                return name
        return "Unknown"

    def check_hand_raised(self, landmarks):
        """
        Check if person has raised hand using MediaPipe pose landmarks
        """
        if not landmarks:
            return False
        
        # Get relevant landmarks
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Check if either hand is above shoulders
        left_raised = left_wrist.y < left_shoulder.y and left_wrist.visibility > 0.5
        right_raised = right_wrist.y < right_shoulder.y and right_wrist.visibility > 0.5
        
        return left_raised or right_raised

    def draw_hand_raise_history(self, frame):
        """
        Draw hand raise history in top right corner
        """
        if self.hand_raise_history:
            padding = 10
            line_height = 30
            total_height = len(self.hand_raise_history) * line_height + 2 * padding
            
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (frame.shape[1] - 300, 0),
                         (frame.shape[1], total_height),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            cv2.putText(frame, "Recent Hand Raises:", 
                       (frame.shape[1] - 290, padding + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i, entry in enumerate(self.hand_raise_history):
                y_pos = padding + (i + 2) * line_height
                cv2.putText(frame, entry,
                           (frame.shape[1] - 290, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def process_video(self):
        frame_count = 0
        start_time = time.time()
        last_hand_raise_time = {}
        
        print("Starting video processing...")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = frame_count / self.fps
            
            # Detect people and their poses
            detected_people = self.detect_people_and_poses(frame)
            
            # Process each detected person
            for person in detected_people:
                # Draw bounding box and name
                box = person['box']
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, person['name'], 
                          (box[0], box[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check for hand raises
                if self.check_hand_raised(person['pose_landmarks']):
                    name = person['name']
                    if (name not in last_hand_raise_time or 
                        current_time - last_hand_raise_time[name] > 2):
                        
                        minutes = int(current_time // 60)
                        seconds = int(current_time % 60)
                        timestamp = f"{minutes:02d}:{seconds:02d}"
                        entry = f"{name} at {timestamp}"
                        
                        self.hand_raise_history.appendleft(entry)
                        print(f"\n🖐 {entry}")
                        last_hand_raise_time[name] = current_time
            
            # Draw hand raise history
            self.draw_hand_raise_history(frame)
            
            # Write frame
            self.writer.write(frame)
            
            # Show progress
            if frame_count % self.fps == 0:
                elapsed_time = time.time() - start_time
                print(f"Processing: {frame_count/self.fps:.1f} seconds...")
            
            # Display frame
            cv2.imshow('Multi-Person Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    tracker = MultiPersonTracker('./desk_video.mp4')
    tracker.process_video()