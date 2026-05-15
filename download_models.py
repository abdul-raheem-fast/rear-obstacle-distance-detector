import urllib.request
import os
import sys


def download_file(url, filename, description):
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\nDownload complete: {filename}")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def download_mobilenet_ssd():
    print("=" * 60)
    print("MobileNet SSD Model Downloader")
    print("=" * 60)
    
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    prototxt_url_alt = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/MobileNetSSD_deploy.prototxt"
    
    prototxt_file = "MobileNetSSD_deploy.prototxt"
    caffemodel_file = "MobileNetSSD_deploy.caffemodel"
    
    success = True
    
    if not os.path.exists(prototxt_file):
        if not download_file(prototxt_url, prototxt_file, "MobileNet SSD config"):
            print("Trying alternative URL...")
            if not download_file(prototxt_url_alt, prototxt_file, "MobileNet SSD config"):
                success = False
    else:
        print(f"{prototxt_file} already exists, skipping download")
    
    if not os.path.exists(caffemodel_file):
        print("\nCaffeModel file needs to be downloaded manually")
        print("Due to Google Drive restrictions, automatic download may fail")
        print("\nPlease download the model manually:")
        print("1. Visit: https://drive.google.com/file/d/0B3MsW3Z1J_3sckVzX0dIU3VpbUE/view")
        print("2. Click 'Download' button")
        print("3. Save as: MobileNetSSD_deploy.caffemodel")
        print("4. Place in this directory")
    else:
        print(f"{caffemodel_file} already exists, skipping download")
    
    return success


def create_sample_calibration_image():
    print("\nCreating sample calibration reference...")
    
    try:
        import cv2
        import numpy as np
        
        width, height = 800, 600
        checkerboard = np.zeros((height, width, 3), dtype=np.uint8)
        
        square_size = 100
        for i in range(0, width, square_size):
            for j in range(0, height, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    checkerboard[j:j+square_size, i:i+square_size] = 255
        
        cv2.putText(checkerboard, "CALIBRATION PATTERN", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(checkerboard, "Use a real object for accurate calibration", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite("calibration_pattern.png", checkerboard)
        print("Created calibration_pattern.png")
        
    except ImportError:
        print("OpenCV not available, skipping calibration pattern creation")


def main():
    print("\n" + "=" * 60)
    print("Reverse Parking Assistant - Model Setup")
    print("=" * 60)
    
    mobilenet_success = download_mobilenet_ssd()
    create_sample_calibration_image()
    
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    files_to_check = [
        "MobileNetSSD_deploy.prototxt",
        "MobileNetSSD_deploy.caffemodel"
    ]
    
    all_ready = True
    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)
            print(f"OK: {filename} ({size:.2f} MB)")
        else:
            print(f"Missing: {filename}")
            all_ready = False
    
    if all_ready:
        print("\nAll model files ready!")
        print("You can now run: python main.py")
    else:
        print("\nSome files are missing")
        print("The application will use fallback motion detection")
    
    print("=" * 60)


if __name__ == "__main__":
    main()