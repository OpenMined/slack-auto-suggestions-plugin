#!/usr/bin/env python3
from PIL import Image, ImageDraw

def create_robot_icon(size):
    # Create a new image with a blue background
    img = Image.new('RGBA', (size, size), (74, 144, 226, 255))
    draw = ImageDraw.Draw(img)
    
    # Scale factor
    scale = size / 20
    
    # White color for robot
    white = (255, 255, 255, 255)
    blue = (74, 144, 226, 255)
    
    # Draw robot head
    draw.rectangle([6*scale, 3*scale, 14*scale, 9*scale], fill=white)
    
    # Draw eyes
    draw.rectangle([8*scale, 5*scale, 10*scale, 7*scale], fill=blue)
    draw.rectangle([12*scale, 5*scale, 14*scale, 7*scale], fill=blue)
    
    # Draw body
    draw.rectangle([5*scale, 10*scale, 15*scale, 17*scale], fill=white)
    
    # Draw arms
    draw.rectangle([3*scale, 11*scale, 5*scale, 15*scale], fill=white)
    draw.rectangle([15*scale, 11*scale, 17*scale, 15*scale], fill=white)
    
    # Draw antenna
    draw.line([10*scale, 3*scale, 10*scale, 1*scale], fill=white, width=int(scale))
    draw.ellipse([9*scale, 0, 11*scale, 2*scale], fill=white)
    
    return img

# Create icons in different sizes
sizes = {
    16: 'robot16.png',
    48: 'robot48.png',
    128: 'robot128.png'
}

for size, filename in sizes.items():
    icon = create_robot_icon(size)
    icon.save(filename)
    print(f"Created {filename}")

print("All icons created successfully!")