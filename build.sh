#!/usr/bin/env bash

set -o errexit

echo "📁 Current dir: $(pwd)"

# Install dependencies
pip install -r requirements.txt

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Move into backend (important for Django)
cd backend

# Collect static files
python manage.py collectstatic --noinput

# Run migrations
python manage.py migrate

echo "✅ Build completed successfully!"