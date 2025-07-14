#!/usr/bin/env python3
"""
Direct Activity Recognition
Runs YOLO person detection + X-CLIP action recognition
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor
import time
from collections import defaultdict, deque
import threading

def crop_and_pad(frame, box, margin=0.15, size=224):
    """Crop person from frame with margin and pad to square"""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(frame.shape[1], x2 + mx), min(frame.shape[0], y2 + my)
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    pad = abs(h - w) // 2
    if h > w:
        crop = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        crop = cv2.copyMakeBorder(crop, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop

def start_activity_recognition():
    """Start webcam activity recognition"""
    
    print("🚀 Starting activity recognition...")
    print("📹 Loading models...")
    
    # Configuration
    YOLO_MODEL = "yolo11s.pt"
    ACTION_MODEL = "microsoft/xclip-base-patch32"
    ACTION_LABELS = [
        "walking", "running", "sitting", "standing", "dancing", 
        "jumping", "waving", "clapping", "lying down", "cooking",
        "reading", "writing", "typing", "exercising", "stretching"
    ]
    SEQUENCE_LENGTH = 8
    CROP_SIZE = 224
    CROP_MARGIN = 0.15
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load YOLO model
    try:
        yolo = YOLO(YOLO_MODEL)
        yolo.to(device)
        print("✅ YOLO model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    # Load action recognition model
    try:
        processor = AutoProcessor.from_pretrained(ACTION_MODEL)
        model = AutoModel.from_pretrained(ACTION_MODEL).to(device).eval()
        print("✅ Action recognition model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading action model: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("📹 Webcam opened successfully!")
    print("👤 Detecting persons and recognizing actions")
    print("🔴 Press 'q' to quit")
    
    # Tracking variables
    track_history = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    last_action = {}
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        frame_count += 1
        
        # Run YOLO detection on person class only
        results = yolo.track(frame, persist=True, classes=[0], conf=0.5)
        
        # Process tracking results
        if results[0].boxes is not None and results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            # Process each tracked person
            for box, tid in zip(boxes, track_ids):
                tid = int(tid)
                
                # Crop person from frame
                crop = crop_and_pad(frame, box, margin=CROP_MARGIN, size=CROP_SIZE)
                track_history[tid].append(crop)
                
                # Run action recognition when we have enough frames
                if len(track_history[tid]) == SEQUENCE_LENGTH and frame_count % 10 == 0:  # Every 10 frames
                    try:
                        # Prepare frames for model
                        processed = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in list(track_history[tid])]
                        
                        # Run inference
                        inputs = processor(
                            videos=[processed],
                            text=ACTION_LABELS,
                            return_tensors="pt",
                            padding=True
                        ).to(device)
                        
                        with torch.inference_mode():
                            outputs = model(**inputs)
                            logits = outputs.logits_per_video[0]
                            probs = logits.softmax(dim=-1)
                            top_idx = probs.argmax().item()
                            confidence = float(probs[top_idx])
                            
                            if confidence > 0.3:  # Only show high confidence actions
                                last_action[tid] = (ACTION_LABELS[top_idx], confidence)
                    
                    except Exception as e:
                        print(f"⚠️  Action recognition error: {e}")
        
        # Draw results
        annotator = cv2.imshow.__self__ if hasattr(cv2, 'imshow') else None
        
        # Draw bounding boxes and action labels
        if results[0].boxes is not None and results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, tid in zip(boxes, track_ids):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID
                label_str = f"ID {tid}"
                
                # Add action if available
                if tid in last_action:
                    action, conf = last_action[tid]
                    label_str += f" | {action} ({conf:.2f})"
                
                # Draw label
                cv2.putText(frame, label_str, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        # Add info text
        cv2.putText(frame, "Activity Recognition - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Activity Recognition", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Activity recognition stopped")

if __name__ == "__main__":
    try:
        start_activity_recognition()
    except KeyboardInterrupt:
        print("\n🛑 Activity recognition stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}") 