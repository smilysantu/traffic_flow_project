Traffic Flow Analysis

Objective:

Develop a Python script to analyze traffic flow by detecting and tracking vehicles in three distinct lanes. The system should count vehicles lane-wise as they cross a defined line and produce both an annotated video and a CSV log.

Approach:

Used YOLOv8 (pre-trained on COCO dataset) for vehicle detection.

Integrated OpenCV + SORT tracker to maintain unique IDs across frames.

Divided the frame into three lanes and added a virtual counting line.

Logged each vehicle crossing into a CSV file with vehicle ID, lane, frame, and timestamp.

Project Structure

traffic_flow_project/
│
├── traffic_flow_analysis.py   # Main script
├── lanes_config.json          # Lane configuration
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
└── outputs/
    ├── output_annotated.mp4   # Annotated sample video
    └── vehicle_counts.csv     # Lane-wise vehicle log

Setup Instructions

1. Clone the repository:

git clone https://github.com/your-username/traffic-flow-analysis.git
cd traffic-flow-analysis


2. Create and activate virtual environment:

python -m venv .venv
.venv\Scripts\activate    # Windows


3. Install dependencies:

pip install -r requirements.txt


4. Run the script:

python traffic_flow_analysis.py



Outputs

output_annotated.mp4: Processed video with lane dividers, bounding boxes, and real-time counts.

vehicle_counts.csv: Structured log file of vehicle entries.


Example CSV format:

vehicle_id	lane	frame	timestamp_seconds

12	1	234	7.8
45	2	480	16.0


Challenges & Solutions

Codec issues: Used XVID/VLC for compatibility.

Lane clarity: Applied distinct colored dividers.

Duplicate counting: Ensured stable IDs with tracker integration.


Outcome

A working traffic analysis system capable of lane-wise vehicle counting, providing both visual and tabular outputs.

Author
Santosh Bhoopali
