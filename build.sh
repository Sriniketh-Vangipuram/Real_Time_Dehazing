#!/usr/bin/env bash

set -o errexit

echo "📁 Current dir: $(pwd)"

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Move into backend (important for Django)
cd backend

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"


# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput



echo "✅ Build completed successfully!"