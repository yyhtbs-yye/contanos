import cv2
import os
import time

def capture_rtsp_to_png():
    # RTSP stream URL
    rtsp_url = "rtsp://192.168.200.206:8554/outstream?tcp"
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return
    
    frame_count = 0
    
    try:
        while True:
            # Read frame from stream
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Generate filename with timestamp
            timestamp = int(time.time() * 1000)  # milliseconds
            filename = f"frame_{frame_count:06d}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame as PNG
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")
            
            frame_count += 1
            
            # Optional: Add delay between captures (adjust as needed)
            time.sleep(0.1)  # Capture every 100ms
            
    except KeyboardInterrupt:
        print("\nCapture stopped by user")
    
    finally:
        # Release resources
        cap.release()
        print(f"Total frames captured: {frame_count}")

if __name__ == "__main__":
    capture_rtsp_to_png()