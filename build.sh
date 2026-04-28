#!/usr/bin/env bash

# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Download YOLO model (if not already present)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Collect static files
python backend/manage.py collectstatic --noinput

# Run migrations
python backend/manage.py migrate

echo "Build completed successfully!"