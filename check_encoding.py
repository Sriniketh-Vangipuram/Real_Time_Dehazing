# check_encoding.py
import chardet

def check_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result

# Check your video.html
video_path = 'core/templates/core/video.html'
try:
    encoding_info = check_file_encoding(video_path)
    print(f"File: {video_path}")
    print(f"Encoding: {encoding_info['encoding']}")
    print(f"Confidence: {encoding_info['confidence']}")
except Exception as e:
    print(f"Error checking {video_path}: {e}")