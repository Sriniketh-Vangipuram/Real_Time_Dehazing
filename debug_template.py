# debug_template_correct.py
import os

# Try to find the correct path
possible_paths = [
    'backend/core/templates/core/video.html',
    'core/templates/core/video.html',
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Found at: {path}")
        print(f"   Absolute path: {os.path.abspath(path)}")
        
        # Read and display content
        with open(path, 'rb') as f:
            content = f.read()
            print(f"\nFile size: {len(content)} bytes")
            
            # Show first 3 lines
            try:
                text = content.decode('utf-8')
                lines = text.split('\n')
                print("First 3 lines:")
                for i, line in enumerate(lines[:3], 1):
                    print(f"  {i}: {repr(line)}")
            except:
                print("Cannot decode as UTF-8")
        break
else:
    print("❌ File not found in any expected location")
    
    # Show directory structure
    print("\nCurrent directory structure:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.html'):
                print(f"{subindent}{file}")