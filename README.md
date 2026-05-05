# PerfLab — AI Athlete Movement Analysis Blog

A Flask + MediaPipe web app that analyses exercise videos frame-by-frame
and returns real joint angles, rep counts, form scores, and annotated frames.

## 6 Projects
1. Pushups        — elbow angle per rep, spine alignment, chest depth
2. Overhead Squat — knee/hip/ankle angles, depth, trunk uprightness
3. Broad Jump     — takeoff angle, landing angle, hip extension, phase snapshots
4. Plank          — spine angle, hip stability, hold duration
5. Pull-ups       — elbow angle per rep, shoulder engagement, symmetry
6. Sprint         — knee drive angle, hip angle, stride asymmetry

## Setup

### 1. Clone / unzip the project
```
cd athlete_app
```

### 2. Create virtual environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the app
```
python app.py
```

### 5. Open browser
```
http://127.0.0.1:5000
```

## Dependencies Used
- **Flask** — web framework / backend server
- **MediaPipe** — Google's pose estimation (33 body landmarks)
- **NumPy** — angle calculations and statistics
- **OpenCV (cv2)** — video reading, frame drawing, image encoding

## How It Works
1. User uploads a video or image via the blog UI
2. Flask receives the file and calls the correct analyser function
3. MediaPipe Pose runs on every frame — extracts 33 3D landmarks
4. NumPy calculates joint angles using the cosine rule
5. Rep counting uses state-machine logic (up/down stage transitions)
6. Annotated frames (with angles overlaid) are encoded as base64 JPEGs
7. Results (metrics, per-rep table, snapshots) are returned as JSON
8. The frontend renders metric cards, findings, per-rep table, and frame grid
