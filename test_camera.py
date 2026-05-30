import cv2
import time

print("=" * 50)
print("Camera Detection Tool")
print("=" * 50)

# Try different backends and camera indices
backends = [
    (cv2.CAP_DSHOW, "DirectShow (Windows)"),
    (cv2.CAP_MSMF, "Media Foundation"),
    (None, "Default")
]

found_any = False

for backend_info in backends:
    backend_id = backend_info[0]
    backend_name = backend_info[1]
    print(f"\nTrying {backend_name}...")
    
    for camera_id in range(5):  # Try cameras 0-4
        try:
            if backend_id is None:
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, backend_id)
            
            time.sleep(0.2)  # Give it time to open
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    found_any = True
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"  ✓ Camera {camera_id}: WORKING!")
                    print(f"    Resolution: {width:.0f}x{height:.0f}")
                    print(f"    FPS: {fps:.1f}")
                else:
                    print(f"  ✗ Camera {camera_id}: Device found but can't capture")
            
            cap.release()
        except Exception as e:
            pass  # Silent fail

print("\n" + "=" * 50)
if found_any:
    print("✓ Found working cameras!")
    print("Note the camera number and update main.py")
else:
    print("✗ No cameras detected!")
    print("\nTroubleshooting steps:")
    print("1. Check Device Manager > Imaging Devices for your webcam")
    print("2. Make sure no other app (Zoom, Teams, OBS) is using the camera")
    print("3. Settings > Privacy & Security > Camera - allow Python access")
    print("4. Restart your computer")
    print("5. If USB camera: try different USB port")
print("=" * 50)
