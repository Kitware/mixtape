#!/usr/bin/env python3
"""
Convert JPEG images embedded in episode JSON to PNG format.
This should improve browser rendering performance.
"""

import json
import base64
import sys
from io import BytesIO
from PIL import Image

def convert_images_to_png(input_file, output_file):
    """Convert all embedded JPEG images to PNG format."""
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_steps = len(data['inference']['steps'])
    converted = 0
    
    print(f"Processing {total_steps} steps...")
    
    for i, step in enumerate(data['inference']['steps']):
        if 'image' in step:
            # Decode the base64 image
            img_data = base64.b64decode(step['image'])
            
            # Open with PIL
            img = Image.open(BytesIO(img_data))
            
            # Convert to PNG
            png_buffer = BytesIO()
            img.save(png_buffer, format='PNG')
            png_buffer.seek(0)
            
            # Re-encode to base64
            step['image'] = base64.b64encode(png_buffer.read()).decode('ascii')
            
            converted += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{total_steps} steps...")
    
    print(f"\nConverted {converted} images from JPEG to PNG")
    print(f"Writing to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Done! Converted file saved to {output_file}")
    print(f"\nTo ingest, run:")
    print(f'docker compose run --rm -v "$PWD:/data" django ./manage.py ingest_episode /data/{output_file}')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_png.json')
    else:
        input_file = 'episode_jpeg_480w.json'
        output_file = 'episode_jpeg_480w_png.json'
    
    convert_images_to_png(input_file, output_file)
