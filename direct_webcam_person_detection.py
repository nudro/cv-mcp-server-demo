#!/usr/bin/env python3
"""
Direct Webcam Person Detection
Runs YOLO detection directly for person class only
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

def start_person_detection():
    """Start webcam detection for person class only"""
    
    print("🚀 Starting direct webcam person detection...")
    print("📹 Loading YOLO model...")
    
    # Load YOLO model
    try:
        model = YOLO("yolo11s.pt")  # or "yolo11n.pt" for faster inference
        print("✅ YOLO model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("📹 Webcam opened successfully!")
    print("👤 Detecting only person class (class 0)")
    print("🔴 Press 'q' to quit")
    
    # COCO class names (person is index 0)
    class_names = ["person"]
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        # Run detection on person class only
        results = model(frame, classes=[0], conf=0.5)  # class 0 = person
        
        # Process results
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Draw detections
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        # Add info text
        cv2.putText(frame, "Person Detection - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Person Detection", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped")

if __name__ == "__main__":
    try:
        start_person_detection()
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}") 