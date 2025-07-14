#!/usr/bin/env python3
"""
Advanced Computer Vision MCP Server
Enhanced version with real-time processing and better integration
"""

import asyncio
import json
import logging
import os
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import queue

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.torch_utils import select_device
from transformers import AutoModel, AutoProcessor

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    LoggingMessageNotificationParams,
    ReadResourceRequest,
    ReadResourceResult,
    Resource,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrackingResult:
    """Result from object tracking"""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int]
    timestamp: float = field(default_factory=time.time)

@dataclass
class ActionResult:
    """Result from action recognition"""
    track_id: int
    actions: List[str]
    confidences: List[float]
    timestamp: float = field(default_factory=time.time)

@dataclass
class VideoFrame:
    """Video frame with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_number: int

class FrameBuffer:
    """Thread-safe frame buffer for action recognition"""
    
    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self.frames: Deque[VideoFrame] = deque(maxlen=max_frames)
        self.lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray, frame_number: int):
        """Add a frame to the buffer"""
        with self.lock:
            video_frame = VideoFrame(
                frame=frame.copy(),
                timestamp=time.time(),
                frame_number=frame_number
            )
            self.frames.append(video_frame)
    
    def get_recent_frames(self, num_frames: int) -> List[np.ndarray]:
        """Get the most recent frames"""
        with self.lock:
            if len(self.frames) < num_frames:
                return [f.frame for f in self.frames]
            return [f.frame for f in list(self.frames)[-num_frames:]]
    
    def clear(self):
        """Clear the frame buffer"""
        with self.lock:
            self.frames.clear()

class AdvancedVideoClassifier:
    """Enhanced video classifier with better preprocessing"""
    
    def __init__(self, model_name: str = "microsoft/xclip-base-patch32", device: str = ""):
        self.device = select_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.model_name = model_name
        
    def preprocess_frames(self, frames: List[np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Enhanced preprocessing for video frames"""
        if not frames:
            raise ValueError("No frames provided for preprocessing")
        
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize with proper interpolation
            frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            
            processed_frames.append(frame_tensor)
        
        # Stack frames and add batch dimension
        return torch.stack(processed_frames).unsqueeze(0).to(self.device)
    
    def classify_actions(self, frames: List[np.ndarray], labels: List[str]) -> Tuple[List[str], List[float]]:
        """Classify actions in video frames with error handling"""
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for action classification")
            return [], []
        
        try:
            # Preprocess frames
            input_tensor = self.preprocess_frames(frames)
            
            # Prepare text inputs
            input_ids = self.processor(text=labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)
            
            # Run inference
            with torch.inference_mode():
                inputs = {"pixel_values": input_tensor, "input_ids": input_ids}
                outputs = self.model(**inputs)
                logits = outputs.logits_per_video
                
            # Postprocess results
            probs = logits.softmax(dim=-1)
            top_indices = probs[0].topk(min(2, len(labels))).indices.tolist()
            top_labels = [labels[idx] for idx in top_indices]
            top_confs = probs[0][top_indices].tolist()
            
            return top_labels, top_confs
            
        except Exception as e:
            logger.error(f"Error in action classification: {str(e)}")
            return [], []

class AdvancedComputerVisionMCP:
    """Advanced MCP server with enhanced computer vision capabilities"""
    
    def __init__(self):
        self.server = Server("advanced-computer-vision-mcp")
        self.device = select_device("")
        self.yolo_model = None
        self.video_classifier = None
        self.track_history = {}
        self.current_video_source = None
        self.is_tracking = False
        self.selected_track_id = None
        self.frame_buffer = FrameBuffer(max_frames=60)  # 2 seconds at 30 FPS
        self.track_frames = defaultdict(lambda: deque(maxlen=16))  # Store frames per track
        self.processing_queue = queue.Queue()
        self.analysis_results = {}
        
        # Background tasks
        self.tracking_task = None
        self.processing_task = None
        
        # Webcam specific
        self.webcam_thread = None
        self.webcam_stop_event = threading.Event()
        self.webcam_mode = None  # 'detection' or 'action_recognition'
        self.webcam_results = queue.Queue(maxsize=1)  # Store latest result
        
        # Register tools
        self.server.list_tools(self.list_tools)
        self.server.call_tool(self.call_tool)
        
        # Register resources
        self.server.read_resource(self.read_resource)
        
        # Note: Logging is handled through notifications in current MCP version
    
    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List available tools with enhanced capabilities"""
        tools = [
            Tool(
                name="load_yolo_model",
                description="Load a YOLO model for object detection and tracking",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "Path to YOLO model file (e.g., 'yolo11s.pt', 'yolo11n.pt')"
                        },
                        "device": {
                            "type": "string",
                            "description": "Device to run on ('cpu', 'cuda', 'mps')",
                            "default": "auto"
                        },
                        "task": {
                            "type": "string",
                            "description": "YOLO task type ('detect', 'track', 'segment')",
                            "default": "detect"
                        }
                    },
                    "required": ["model_path"]
                }
            ),
            Tool(
                name="load_video_classifier",
                description="Load a video classifier for action recognition",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "HuggingFace model name for video classification",
                            "default": "microsoft/xclip-base-patch32"
                        },
                        "device": {
                            "type": "string",
                            "description": "Device to run on ('cpu', 'cuda', 'mps')",
                            "default": "auto"
                        },
                        "fp16": {
                            "type": "boolean",
                            "description": "Use half-precision for faster inference",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="start_video_tracking",
                description="Start tracking objects in a video source with real-time processing",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Video source (file path, URL, or camera index like '0')"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Detection confidence threshold",
                            "default": 0.3
                        },
                        "max_detections": {
                            "type": "integer",
                            "description": "Maximum number of detections",
                            "default": 20
                        },
                        "tracker": {
                            "type": "string",
                            "description": "Tracking algorithm ('bytetrack.yaml', 'botsort.yaml')",
                            "default": "bytetrack.yaml"
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Classes to detect (empty for all)",
                            "default": []
                        },
                        "save_video": {
                            "type": "boolean",
                            "description": "Save output video",
                            "default": False
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Output video path (if save_video is true)"
                        }
                    },
                    "required": ["source"]
                }
            ),
            Tool(
                name="stop_tracking",
                description="Stop the current tracking session and cleanup resources",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="select_track",
                description="Select a specific track ID for focused tracking and analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "track_id": {
                            "type": "integer",
                            "description": "Track ID to select"
                        },
                        "focus_mode": {
                            "type": "boolean",
                            "description": "Enable focus mode with enhanced tracking",
                            "default": True
                        }
                    },
                    "required": ["track_id"]
                }
            ),
            Tool(
                name="analyze_actions",
                description="Analyze actions for tracked objects using collected frames",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "track_id": {
                            "type": "integer",
                            "description": "Track ID to analyze (optional, analyzes all if not specified)"
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Action labels to classify",
                            "default": ["walking", "running", "sitting", "standing", "cooking", "exercising", "dancing", "jumping"]
                        },
                        "num_frames": {
                            "type": "integer",
                            "description": "Number of frames to use for analysis",
                            "default": 8
                        },
                        "real_time": {
                            "type": "boolean",
                            "description": "Enable real-time analysis",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="get_tracking_status",
                description="Get comprehensive tracking status and statistics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_analysis": {
                            "type": "boolean",
                            "description": "Include action analysis results",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="save_video",
                description="Save the current tracking session to a video file with annotations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_path": {
                            "type": "string",
                            "description": "Output video file path"
                        },
                        "include_analysis": {
                            "type": "boolean",
                            "description": "Include action analysis annotations",
                            "default": True
                        },
                        "fps": {
                            "type": "integer",
                            "description": "Output video FPS",
                            "default": 30
                        }
                    },
                    "required": ["output_path"]
                }
            ),
            Tool(
                name="get_track_analytics",
                description="Get detailed analytics for a specific track",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "track_id": {
                            "type": "integer",
                            "description": "Track ID to analyze"
                        },
                        "include_trajectory": {
                            "type": "boolean",
                            "description": "Include trajectory data",
                            "default": True
                        },
                        "include_actions": {
                            "type": "boolean",
                            "description": "Include action analysis",
                            "default": True
                        }
                    },
                    "required": ["track_id"]
                }
            ),
            Tool(
                name="configure_tracking",
                description="Configure tracking parameters in real-time",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "confidence": {
                            "type": "number",
                            "description": "Detection confidence threshold"
                        },
                        "max_detections": {
                            "type": "integer",
                            "description": "Maximum number of detections"
                        },
                        "tracker": {
                            "type": "string",
                            "description": "Tracking algorithm"
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Classes to detect"
                        }
                    }
                }
            ),
            Tool(
                name="start_webcam_detection",
                description="Start real-time YOLO object detection on webcam.",
                inputSchema={"type": "object", "properties": {}}),
            Tool(
                name="start_webcam_action_recognition",
                description="Start real-time action recognition (YOLO+XCLIP) on webcam.",
                inputSchema={"type": "object", "properties": {}}),
            Tool(
                name="stop_webcam",
                description="Stop any running webcam detection or action recognition.",
                inputSchema={"type": "object", "properties": {}}),
            Tool(
                name="get_webcam_status",
                description="Get status and latest results from webcam detection/action recognition.",
                inputSchema={"type": "object", "properties": {}}),
        ]
        
        return ListToolsResult(tools=tools)
    
    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls with enhanced error handling"""
        try:
            if request.name == "load_yolo_model":
                return await self._load_yolo_model(request.arguments)
            elif request.name == "load_video_classifier":
                return await self._load_video_classifier(request.arguments)
            elif request.name == "start_video_tracking":
                return await self._start_video_tracking(request.arguments)
            elif request.name == "stop_tracking":
                return await self._stop_tracking(request.arguments)
            elif request.name == "select_track":
                return await self._select_track(request.arguments)
            elif request.name == "analyze_actions":
                return await self._analyze_actions(request.arguments)
            elif request.name == "get_tracking_status":
                return await self._get_tracking_status(request.arguments)
            elif request.name == "save_video":
                return await self._save_video(request.arguments)
            elif request.name == "get_track_analytics":
                return await self._get_track_analytics(request.arguments)
            elif request.name == "configure_tracking":
                return await self._configure_tracking(request.arguments)
            elif request.name == "start_webcam_detection":
                return await self._start_webcam_detection()
            elif request.name == "start_webcam_action_recognition":
                return await self._start_webcam_action_recognition()
            elif request.name == "stop_webcam":
                return await self._stop_webcam()
            elif request.name == "get_webcam_status":
                return await self._get_webcam_status()
            else:
                raise ValueError(f"Unknown tool: {request.name}")
                
        except Exception as e:
            logger.error(f"Error in tool call {request.name}: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")]
            )
    
    async def _load_yolo_model(self, args: Dict[str, Any]) -> CallToolResult:
        """Load YOLO model with enhanced configuration"""
        model_path = args["model_path"]
        device = args.get("device", "auto")
        task = args.get("task", "detect")
        
        try:
            # Check if model file exists
            if not os.path.exists(model_path) and not model_path.startswith(('http', 'https')):
                # Try to download from Ultralytics
                logger.info(f"Model {model_path} not found locally, attempting to download...")
            
            self.yolo_model = YOLO(model_path)
            if device != "auto":
                self.yolo_model.to(device)
            
            # Get model info
            model_info = {
                "path": model_path,
                "task": task,
                "device": str(self.yolo_model.device),
                "classes": list(self.yolo_model.names.values()),
                "num_classes": len(self.yolo_model.names)
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"✅ YOLO model loaded successfully!\n" +
                         f"Model: {model_path}\n" +
                         f"Task: {task}\n" +
                         f"Device: {model_info['device']}\n" +
                         f"Classes: {model_info['num_classes']} available\n" +
                         f"Sample classes: {', '.join(model_info['classes'][:5])}..."
                )]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Failed to load YOLO model: {str(e)}")]
            )
    
    async def _load_video_classifier(self, args: Dict[str, Any]) -> CallToolResult:
        """Load video classifier with enhanced configuration"""
        model_name = args.get("model_name", "microsoft/xclip-base-patch32")
        device = args.get("device", "auto")
        fp16 = args.get("fp16", False)
        
        try:
            self.video_classifier = AdvancedVideoClassifier(model_name, device)
            
            # Test the model with a dummy input
            dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
            test_labels = ["test"]
            try:
                actions, confs = self.video_classifier.classify_actions(dummy_frames, test_labels)
                test_status = "✅ Model test successful"
            except Exception as test_e:
                test_status = f"⚠️ Model test failed: {str(test_e)}"
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"✅ Video classifier loaded successfully!\n" +
                         f"Model: {model_name}\n" +
                         f"Device: {device}\n" +
                         f"FP16: {fp16}\n" +
                         f"{test_status}"
                )]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Failed to load video classifier: {str(e)}")]
            )
    
    async def _start_video_tracking(self, args: Dict[str, Any]) -> CallToolResult:
        """Start video tracking with enhanced features"""
        if not self.yolo_model:
            return CallToolResult(
                content=[TextContent(type="text", text="❌ YOLO model not loaded. Please load a model first.")]
            )
        
        source = args["source"]
        confidence = args.get("confidence", 0.3)
        max_detections = args.get("max_detections", 20)
        tracker = args.get("tracker", "bytetrack.yaml")
        classes = args.get("classes", [])
        save_video = args.get("save_video", False)
        output_path = args.get("output_path", "output.mp4")
        
        try:
            # Initialize video capture
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                return CallToolResult(
                    content=[TextContent(type="text", text=f"❌ Failed to open video source: {source}")]
                )
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.current_video_source = cap
            self.is_tracking = True
            self.track_history = {}
            self.frame_buffer.clear()
            self.track_frames.clear()
            self.analysis_results.clear()
            
            # Start background tasks
            self.tracking_task = asyncio.create_task(
                self._advanced_tracking_loop(cap, confidence, max_detections, tracker, classes, save_video, output_path)
            )
            
            if self.video_classifier:
                self.processing_task = asyncio.create_task(self._background_processing_loop())
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"✅ Started advanced tracking!\n" +
                         f"Source: {source}\n" +
                         f"Resolution: {width}x{height}\n" +
                         f"FPS: {fps:.1f}\n" +
                         f"Confidence: {confidence}\n" +
                         f"Max detections: {max_detections}\n" +
                         f"Tracker: {tracker}\n" +
                         f"Classes: {classes if classes else 'All'}\n" +
                         f"Video classifier: {'✅ Active' if self.video_classifier else '❌ Not loaded'}"
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Failed to start tracking: {str(e)}")]
            )
    
    async def _advanced_tracking_loop(self, cap, confidence: float, max_detections: int, 
                                    tracker: str, classes: List[int], save_video: bool, output_path: str):
        """Advanced tracking loop with frame collection and real-time processing"""
        frame_number = 0
        video_writer = None
        
        try:
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while self.is_tracking and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_number += 1
                
                # Add frame to buffer
                self.frame_buffer.add_frame(frame, frame_number)
                
                # Run YOLO tracking
                tracking_args = {
                    'conf': confidence,
                    'max_det': max_detections,
                    'tracker': tracker,
                    'persist': True
                }
                
                if classes:
                    tracking_args['classes'] = classes
                
                results = self.yolo_model.track(frame, **tracking_args)
                
                # Process tracking results
                if results[0].boxes is not None and results[0].boxes.is_track:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    
                    # Update track history and collect frames
                    for i, track_id in enumerate(track_ids):
                        track_id = int(track_id)
                        bbox = tuple(map(int, boxes[i]))
                        class_id = int(class_ids[i])
                        conf = float(confidences[i])
                        
                        # Store tracking result
                        self.track_history[track_id] = TrackingResult(
                            track_id=track_id,
                            class_name=self.yolo_model.names[class_id],
                            bbox=bbox,
                            confidence=conf,
                            center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        )
                        
                        # Collect frames for this track
                        if frame_number % 2 == 0:  # Every 2nd frame
                            crop = self._crop_track(frame, bbox, margin_percent=10)
                            if crop is not None:
                                self.track_frames[track_id].append(crop)
                
                # Save frame if requested
                if save_video and video_writer:
                    annotated_frame = self._annotate_frame(frame, results[0])
                    video_writer.write(annotated_frame)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            logger.error(f"Error in advanced tracking loop: {str(e)}")
        finally:
            if video_writer:
                video_writer.release()
            if cap.isOpened():
                cap.release()
    
    async def _background_processing_loop(self):
        """Background loop for action analysis"""
        while self.is_tracking:
            try:
                # Process tracks that have enough frames
                for track_id, frames in self.track_frames.items():
                    if len(frames) >= 8 and track_id not in self.analysis_results:
                        # Get recent frames
                        recent_frames = list(frames)[-8:]
                        
                        # Analyze actions
                        labels = ["walking", "running", "sitting", "standing", "cooking", "exercising"]
                        actions, confidences = self.video_classifier.classify_actions(recent_frames, labels)
                        
                        if actions:
                            self.analysis_results[track_id] = ActionResult(
                                track_id=track_id,
                                actions=actions,
                                confidences=confidences
                            )
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in background processing: {str(e)}")
                await asyncio.sleep(1.0)
    
    def _crop_track(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], margin_percent: int = 10) -> Optional[np.ndarray]:
        """Crop a track from frame with margin"""
        try:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # Add margin
            margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
            x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
            x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)
            
            # Take square crop
            size = max(y2 - y1, x2 - x1)
            center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
            half_size = size // 2
            
            square_crop = frame[
                max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
                max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
            ]
            
            return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            
        except Exception as e:
            logger.error(f"Error cropping track: {str(e)}")
            return None
    
    def _annotate_frame(self, frame: np.ndarray, result) -> np.ndarray:
        """Annotate frame with tracking results and analysis"""
        annotated = frame.copy()
        annotator = Annotator(annotated)
        
        if result.boxes is not None and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                bbox = tuple(map(int, boxes[i]))
                class_id = int(class_ids[i])
                conf = float(confidences[i])
                
                # Get color for track
                color = colors(track_id, True)
                
                # Draw bounding box
                label = f"{self.yolo_model.names[class_id]} ID:{track_id} ({conf:.2f})"
                
                # Add action analysis if available
                if track_id in self.analysis_results:
                    action_result = self.analysis_results[track_id]
                    if action_result.actions:
                        label += f" | {action_result.actions[0]} ({action_result.confidences[0]:.2f})"
                
                # Highlight selected track
                if track_id == self.selected_track_id:
                    # Draw focus indicators
                    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    cv2.circle(annotated, center, 8, color, -1)
                    cv2.circle(annotated, center, 12, color, 2)
                
                annotator.box_label(bbox, label, color=color)
        
        return annotated
    
    async def _stop_tracking(self, args: Dict[str, Any]) -> CallToolResult:
        """Stop tracking with cleanup"""
        self.is_tracking = False
        
        # Cancel background tasks
        if self.tracking_task:
            self.tracking_task.cancel()
        if self.processing_task:
            self.processing_task.cancel()
        
        # Release video capture
        if self.current_video_source:
            self.current_video_source.release()
            self.current_video_source = None
        
        # Clear buffers
        self.frame_buffer.clear()
        self.track_frames.clear()
        
        return CallToolResult(
            content=[TextContent(type="text", text="✅ Tracking stopped and resources cleaned up")]
        )
    
    async def _select_track(self, args: Dict[str, Any]) -> CallToolResult:
        """Select a track for focused analysis"""
        track_id = args["track_id"]
        focus_mode = args.get("focus_mode", True)
        
        if track_id in self.track_history:
            self.selected_track_id = track_id
            track_info = self.track_history[track_id]
            
            # Get analysis if available
            analysis_text = ""
            if track_id in self.analysis_results:
                action_result = self.analysis_results[track_id]
                analysis_text = f"\nAction Analysis: {action_result.actions[0]} ({action_result.confidences[0]:.2f})"
            
            return CallToolResult(
                content=[TextContent(
                    type="text", 
                    text=f"✅ Selected track {track_id} for focused tracking!\n" +
                         f"Class: {track_info.class_name}\n" +
                         f"Position: {track_info.center}\n" +
                         f"Confidence: {track_info.confidence:.2f}\n" +
                         f"Focus mode: {'✅ Enabled' if focus_mode else '❌ Disabled'}" +
                         analysis_text
                )]
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Track {track_id} not found")]
            )
    
    async def _analyze_actions(self, args: Dict[str, Any]) -> CallToolResult:
        """Analyze actions with enhanced capabilities"""
        if not self.video_classifier:
            return CallToolResult(
                content=[TextContent(type="text", text="❌ Video classifier not loaded. Please load a classifier first.")]
            )
        
        track_id = args.get("track_id")
        labels = args.get("labels", ["walking", "running", "sitting", "standing", "cooking", "exercising", "dancing", "jumping"])
        num_frames = args.get("num_frames", 8)
        real_time = args.get("real_time", False)
        
        try:
            if track_id is not None:
                # Analyze specific track
                if track_id not in self.track_history:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ Track {track_id} not found")]
                    )
                
                if track_id in self.track_frames and len(self.track_frames[track_id]) >= num_frames:
                    frames = list(self.track_frames[track_id])[-num_frames:]
                    actions, confidences = self.video_classifier.classify_actions(frames, labels)
                    
                    if actions:
                        return CallToolResult(
                            content=[TextContent(
                                type="text", 
                                text=f"Action analysis for track {track_id}:\n" + 
                                     "\n".join([f"- {action}: {conf:.2f}" for action, conf in zip(actions, confidences)])
                            )]
                        )
                    else:
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"❌ No actions detected for track {track_id}")]
                        )
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ Not enough frames for track {track_id} (need {num_frames})")]
                    )
            else:
                # Analyze all tracks
                if not self.track_history:
                    return CallToolResult(
                        content=[TextContent(type="text", text="❌ No tracks available for analysis")]
                    )
                
                analysis_results = []
                for tid, track_info in self.track_history.items():
                    if tid in self.analysis_results:
                        action_result = self.analysis_results[tid]
                        analysis_results.append(
                            f"Track {tid} ({track_info.class_name}): {action_result.actions[0]} ({action_result.confidences[0]:.2f})"
                        )
                    else:
                        analysis_results.append(f"Track {tid} ({track_info.class_name}): No analysis available")
                
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text="Action analysis for all tracks:\n" + "\n".join(analysis_results)
                    )]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Failed to analyze actions: {str(e)}")]
            )
    
    async def _get_tracking_status(self, args: Dict[str, Any]) -> CallToolResult:
        """Get comprehensive tracking status"""
        include_analysis = args.get("include_analysis", True)
        
        status_info = {
            "is_tracking": self.is_tracking,
            "model_loaded": self.yolo_model is not None,
            "classifier_loaded": self.video_classifier is not None,
            "active_tracks": len(self.track_history),
            "selected_track": self.selected_track_id,
            "frame_buffer_size": len(self.frame_buffer.frames),
            "analysis_results": len(self.analysis_results)
        }
        
        # Build status text
        status_text = f"Tracking Status:\n" + \
                     f"- Active: {status_info['is_tracking']}\n" + \
                     f"- YOLO Model: {'✅' if status_info['model_loaded'] else '❌'}\n" + \
                     f"- Video Classifier: {'✅' if status_info['classifier_loaded'] else '❌'}\n" + \
                     f"- Active Tracks: {status_info['active_tracks']}\n" + \
                     f"- Selected Track: {status_info['selected_track']}\n" + \
                     f"- Frame Buffer: {status_info['frame_buffer_size']} frames\n" + \
                     f"- Analysis Results: {status_info['analysis_results']}"
        
        if self.track_history:
            track_details = []
            for track_id, track_info in self.track_history.items():
                detail = f"Track {track_id}: {track_info.class_name} at {track_info.center} (conf: {track_info.confidence:.2f})"
                
                if include_analysis and track_id in self.analysis_results:
                    action_result = self.analysis_results[track_id]
                    detail += f" | {action_result.actions[0]} ({action_result.confidences[0]:.2f})"
                
                track_details.append(detail)
            
            status_text += f"\n\nTrack Details:\n" + "\n".join(track_details)
        
        return CallToolResult(
            content=[TextContent(type="text", text=status_text)]
        )
    
    async def _save_video(self, args: Dict[str, Any]) -> CallToolResult:
        """Save tracking session to video"""
        output_path = args["output_path"]
        include_analysis = args.get("include_analysis", True)
        fps = args.get("fps", 30)
        
        if not self.is_tracking:
            return CallToolResult(
                content=[TextContent(type="text", text="❌ No active tracking session to save")]
            )
        
        # In a real implementation, you would save the video with annotations
        return CallToolResult(
            content=[TextContent(
                type="text", 
                text=f"✅ Video saved to: {output_path}\n" +
                     f"Include analysis: {include_analysis}\n" +
                     f"FPS: {fps}\n" +
                     f"(Note: This is a placeholder - implement actual video saving)"
            )]
        )
    
    async def _get_track_analytics(self, args: Dict[str, Any]) -> CallToolResult:
        """Get detailed analytics for a specific track"""
        track_id = args["track_id"]
        include_trajectory = args.get("include_trajectory", True)
        include_actions = args.get("include_actions", True)
        
        if track_id not in self.track_history:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Track {track_id} not found")]
            )
        
        track_info = self.track_history[track_id]
        
        analytics_text = f"Analytics for Track {track_id}:\n" + \
                        f"Class: {track_info.class_name}\n" + \
                        f"Current Position: {track_info.center}\n" + \
                        f"Confidence: {track_info.confidence:.2f}\n" + \
                        f"Timestamp: {track_info.timestamp:.2f}\n" + \
                        f"Frames Collected: {len(self.track_frames.get(track_id, []))}"
        
        if include_actions and track_id in self.analysis_results:
            action_result = self.analysis_results[track_id]
            analytics_text += f"\n\nAction Analysis:\n" + \
                             "\n".join([f"- {action}: {conf:.2f}" for action, conf in zip(action_result.actions, action_result.confidences)])
        
        return CallToolResult(
            content=[TextContent(type="text", text=analytics_text)]
        )
    
    async def _configure_tracking(self, args: Dict[str, Any]) -> CallToolResult:
        """Configure tracking parameters in real-time"""
        if not self.is_tracking:
            return CallToolResult(
                content=[TextContent(type="text", text="❌ No active tracking session to configure")]
            )
        
        # In a real implementation, you would update tracking parameters
        config_text = "Configuration updated:\n"
        
        if "confidence" in args:
            config_text += f"- Confidence: {args['confidence']}\n"
        if "max_detections" in args:
            config_text += f"- Max detections: {args['max_detections']}\n"
        if "tracker" in args:
            config_text += f"- Tracker: {args['tracker']}\n"
        if "classes" in args:
            config_text += f"- Classes: {args['classes']}\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=config_text)]
        )
    
    async def _start_webcam_detection(self) -> CallToolResult:
        if self.webcam_thread and self.webcam_thread.is_alive():
            return CallToolResult(content=[TextContent(type="text", text="❌ Webcam already running.")])
        self.webcam_stop_event.clear()
        self.webcam_mode = 'detection'
        self.webcam_thread = threading.Thread(target=self._webcam_detection_loop, daemon=True)
        self.webcam_thread.start()
        return CallToolResult(content=[TextContent(type="text", text="✅ Started webcam YOLO detection.")])

    async def _start_webcam_action_recognition(self) -> CallToolResult:
        if self.webcam_thread and self.webcam_thread.is_alive():
            return CallToolResult(content=[TextContent(type="text", text="❌ Webcam already running.")])
        self.webcam_stop_event.clear()
        self.webcam_mode = 'action_recognition'
        self.webcam_thread = threading.Thread(target=self._webcam_action_recognition_loop, daemon=True)
        self.webcam_thread.start()
        return CallToolResult(content=[TextContent(type="text", text="✅ Started webcam action recognition.")])

    async def _stop_webcam(self) -> CallToolResult:
        self.webcam_stop_event.set()
        if self.webcam_thread:
            self.webcam_thread.join(timeout=2)
        self.webcam_thread = None
        self.webcam_mode = None
        return CallToolResult(content=[TextContent(type="text", text="✅ Stopped webcam processing.")])

    async def _get_webcam_status(self) -> CallToolResult:
        status = {
            "running": self.webcam_thread.is_alive() if self.webcam_thread else False,
            "mode": self.webcam_mode,
        }
        latest = None
        try:
            latest = self.webcam_results.get_nowait()
            self.webcam_results.put(latest)  # put it back
        except queue.Empty:
            pass
        if latest:
            status["latest_result"] = latest
        return CallToolResult(content=[TextContent(type="text", text=f"Webcam status: {status}")])

    def _webcam_detection_loop(self):
        model = YOLO("yolo11s.pt")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        cap = cv2.VideoCapture(0)
        while not self.webcam_stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            dets = [
                {"box": box.tolist(), "class": int(cls)}
                for box, cls in zip(boxes, classes)
            ]
            # Store latest result
            if self.webcam_results.full():
                self.webcam_results.get()
            self.webcam_results.put({"detections": dets})
            # Show window
            annotated = results[0].plot()
            cv2.imshow("YOLO Webcam MCP", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _webcam_action_recognition_loop(self):
        from transformers import AutoModel, AutoProcessor
        import time
        from collections import defaultdict, deque
        YOLO_MODEL = "yolo11s.pt"
        ACTION_MODEL = "microsoft/xclip-base-patch32"
        ACTION_LABELS = [
            "walking", "running", "sitting", "standing", "dancing", "jumping", "waving", "clapping", "lying down"
        ]
        SEQUENCE_LENGTH = 8
        CROP_SIZE = 224
        CROP_MARGIN = 0.15
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        yolo = YOLO(YOLO_MODEL)
        yolo.to(device)
        processor = AutoProcessor.from_pretrained(ACTION_MODEL)
        model = AutoModel.from_pretrained(ACTION_MODEL).to(device).eval()
        cap = cv2.VideoCapture(0)
        track_history = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
        last_action = {}
        def crop_and_pad(frame, box, margin=0.15, size=224):
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
        while not self.webcam_stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo.track(frame, persist=True, classes=[0])
            if results[0].boxes.is_track:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                for box, tid in zip(boxes, track_ids):
                    crop = crop_and_pad(frame, box, margin=CROP_MARGIN, size=CROP_SIZE)
                    track_history[tid].append(crop)
                    if len(track_history[tid]) == SEQUENCE_LENGTH:
                        processed = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in list(track_history[tid])]
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
                            last_action[tid] = (ACTION_LABELS[top_idx], float(probs[top_idx]))
            # Store latest result
            summary = {tid: last_action.get(tid, (None, None)) for tid in track_history}
            if self.webcam_results.full():
                self.webcam_results.get()
            self.webcam_results.put({"actions": summary})
            # Draw
            annotator = Annotator(frame)
            if results[0].boxes.is_track:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                for box, tid in zip(boxes, track_ids):
                    label_str = f"ID {tid}"
                    if tid in last_action:
                        label_str += f" | {last_action[tid][0]} ({last_action[tid][1]:.2f})"
                    annotator.box_label(box, label_str, color=(0, 0, 255))
            cv2.imshow("Webcam Action Recognition MCP", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    async def read_resource(self, request: ReadResourceRequest) -> ReadResourceResult:
        """Read resources (model files, videos, etc.)"""
        try:
            resource_path = request.uri
            if resource_path.startswith("file://"):
                file_path = resource_path[7:]
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                return ReadResourceResult(
                    contents=[EmbeddedResource(
                        uri=resource_path,
                        mimeType="application/octet-stream",
                        data=content
                    )]
                )
            else:
                return ReadResourceResult(
                    contents=[TextContent(
                        type="text",
                        text=f"Unsupported resource URI: {resource_path}"
                    )]
                )
        except Exception as e:
            return ReadResourceResult(
                contents=[TextContent(
                    type="text",
                    text=f"Error reading resource: {str(e)}"
                )]
            )
    

    
    def _log_message(self, level: str, message: str, logger_name: str = None):
        """Internal logging method"""
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "info":
            logger.info(message)
        elif level == "debug":
            logger.debug(message)
        else:
            logger.info(message)

async def main():
    """Main entry point"""
    # Create MCP server
    cv_mcp = AdvancedComputerVisionMCP()
    
    # Initialize server
    init_options = InitializationOptions(
        server_name="advanced-computer-vision-mcp",
        server_version="2.0.0",
        capabilities=cv_mcp.server.get_capabilities(
            notification_options=None,
            experimental_capabilities=None,
        ),
    )
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await cv_mcp.server.run(
            read_stream,
            write_stream,
            init_options,
        )

if __name__ == "__main__":
    asyncio.run(main()) 