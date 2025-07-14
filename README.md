# Computer Vision MCP Demo

A comprehensive computer vision demo showcasing real-time person detection, tracking, and activity recognition using YOLO models. This repo is designed for use with the [GitHub MCP server](https://github.com/nudro/cv-mcp-server-demo) (as provided by Cursor IDE) and as a standalone demo for local computer vision tasks.

## 🚀 Features

- **Real-time Person Detection**: YOLO-based person detection with webcam integration
- **Activity Recognition**: X-CLIP powered action recognition for 15+ activities
- **Object Tracking**: Persistent person tracking across video frames
- **Multiple Demo Scripts**: Standalone scripts for different use cases

## 📋 Prerequisites

- Python 3.8+
- Apple Silicon Mac (M1/M2/M3) for optimal performance
- Webcam access
- Git

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nudro/cv-mcp-server-demo.git
   cd cv-mcp-server-demo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO models** (automatically on first run):
   - YOLO11s.pt (high accuracy)
   - YOLO11n.pt (fast inference)

## 🎯 Quick Start

### Person Detection Only
```bash
python direct_webcam_person_detection.py
```

### Activity Recognition
```bash
python direct_activity_recognition.py
```

### MCP Server (for advanced users)
If you want to run your own MCP server (not required for most Cursor users):
```bash
python cv_mcp_server_advanced.py
```

> **Note:** For most users, the MCP server is provided by Cursor IDE and you do not need to run your own server. Use the local scripts for direct computer vision demos.

## 📁 Project Structure

```
cv-mcp-server-demo/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── cv_mcp_server_advanced.py           # (Optional) Standalone MCP server implementation
├── direct_webcam_person_detection.py   # Person detection demo
├── direct_activity_recognition.py      # Activity recognition demo
├── ultralytics_action_recognition.py   # Ultralytics action recognition example
├── ultralytics_interactive_tracker.py  # Ultralytics interactive tracker example
├── cursor_mcp_config.json              # Cursor IDE MCP configuration
├── mcp_config.json                     # Standard MCP configuration
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore file
```

## 🎮 Usage

### Person Detection
- **Purpose**: Real-time person detection using YOLO
- **Features**: 
  - Person-only detection (class 0)
  - Bounding box visualization
  - Confidence scores
  - Center point tracking
- **Controls**: Press 'q' to quit

### Activity Recognition
- **Purpose**: Detect and classify human activities
- **Supported Actions**:
  - walking, running, sitting, standing, dancing
  - jumping, waving, clapping, lying down, cooking
  - reading, writing, typing, exercising, stretching
- **Features**:
  - Person tracking with unique IDs
  - Action classification with confidence scores
  - Temporal analysis (8-frame sequences)
- **Controls**: Press 'q' to quit

### MCP Server (Advanced)
- **Purpose**: Model Context Protocol server for AI tool integration
- **Features**:
  - Tool registration for computer vision tasks
  - Resource management
  - Async processing capabilities
- **Configuration**: Use provided config files for IDE integration
- **Note**: Most users should use the MCP server provided by Cursor IDE.

## 🔧 Configuration

### Cursor IDE Integration
Copy `cursor_mcp_config.json` to your Cursor configuration directory if you want to use your own server:
```bash
cp cursor_mcp_config.json ~/.cursor/mcp_servers/
```

### Standard MCP Configuration
Use `mcp_config.json` for standard MCP client integration.

## 📊 Performance

- **Detection Speed**: ~15ms per frame (60+ FPS)
- **Action Recognition**: Every 10 frames for efficiency
- **Device**: Optimized for Apple Silicon (MPS)
- **Memory**: Efficient GPU memory usage

## 🛠️ Development

### Adding New Actions
Edit `direct_activity_recognition.py` and modify the `ACTION_LABELS` list:
```python
ACTION_LABELS = [
    "walking", "running", "sitting", "standing", "dancing", 
    "jumping", "waving", "clapping", "lying down", "cooking",
    "reading", "writing", "typing", "exercising", "stretching",
    "your_new_action"  # Add here
]
```

### Customizing Detection
Modify confidence thresholds and detection parameters in the demo scripts:
```python
# Person detection confidence
results = yolo.track(frame, persist=True, classes=[0], conf=0.5)

# Action recognition confidence
if confidence > 0.3:  # Adjust threshold
```

## 🤝 Contributing & Extending

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Extending with MCP Tools
If you want to register your own tools with the GitHub MCP server (as used by Cursor), see the [Cursor documentation](https://www.cursor.so/docs) or [GitHub MCP server docs](https://github.com/nudro/cv-mcp-server-demo) for plugin/tool API details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO models and examples
- **Microsoft**: X-CLIP action recognition model
- **OpenCV**: Computer vision framework
- **PyTorch**: Deep learning framework

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not found**:
   - Ensure webcam permissions are granted
   - Check if another application is using the webcam

2. **Model download issues**:
   - Check internet connection
   - Models download automatically on first run

3. **Performance issues**:
   - Ensure you're using Apple Silicon for optimal performance
   - Close other GPU-intensive applications

4. **MCP server errors**:
   - If using Cursor, ensure the GitHub MCP server is running (usually automatic)
   - If running your own server, check MCP library version compatibility and configuration files

### Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

## 📈 Future Enhancements

- [ ] Multi-person activity recognition
- [ ] Gesture recognition
- [ ] Pose estimation integration
- [ ] Video file processing
- [ ] Real-time analytics dashboard
- [ ] Mobile app integration
- [ ] Cloud deployment support
