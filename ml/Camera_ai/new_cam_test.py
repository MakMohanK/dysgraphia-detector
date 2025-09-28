import cv2
import os
from datetime import datetime

# Folder to save cropped images
OUTPUT_DIR = "snips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mouse drawing state
drawing = False
ix = iy = -1
rx = ry = -1
rect_done = False

def mouse_callback(event, x, y, flags, param):
    global ix, iy, rx, ry, drawing, rect_done, base_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_done = False
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # update current rectangle end for live preview
            rx, ry = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry = x, y
        rect_done = True

        # Crop from the grayscale frame (param expected to be current grayscale)
        gray_frame = param['gray'].copy()
        h, w = gray_frame.shape

        # Normalize coordinates (ensure inside frame)
        x1, x2 = sorted((max(0, min(ix, w-1)), max(0, min(rx, w-1))))
        y1, y2 = sorted((max(0, min(iy, h-1)), max(0, min(ry, h-1))))

        # Avoid zero-size crop
        if x2 - x1 > 5 and y2 - y1 > 5:
            crop = gray_frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(OUTPUT_DIR, f"snip_{timestamp}.png")
            cv2.imwrite(filename, crop)
            print(f"[Saved] {filename}")
        else:
            print("[Info] Selection too small — not saved.")

def live_snip_tool(camera_index=0):
    global ix, iy, rx, ry, drawing, rect_done

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("Grayscale (select to snip)")
    cv2.namedWindow("Original Feed (Rotated)")
    # we'll pass current gray to the mouse callback via a dict (param)
    callback_param = {'gray': None}
    cv2.setMouseCallback("Grayscale (select to snip)", mouse_callback, callback_param)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: empty frame.")
            break

        # Rotate camera feed by 180 degrees (if required)
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding for better text extraction
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 11
        )

        # Invert black and white
        bw_inverted = cv2.bitwise_not(bw)

        # Update param for callback
        callback_param['gray'] = bw_inverted

        # Create a display copy for drawing the rectangle overlay
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # If currently drawing or rect_done, draw rectangle overlay
        if drawing or rect_done:
            x1, y1 = ix, iy
            x2, y2 = rx, ry
            # protect against -1 initial values
            if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Show windows
        cv2.imshow("Original Feed (Rotated)", frame)
        cv2.imshow("Grayscale (select to snip)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # clear rectangle
            ix = iy = rx = ry = -1
            drawing = rect_done = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Instructions:")
    print(" - Click and drag on the 'Grayscale (select to snip)' window to select an area.")
    print(" - On mouse release, the selected grayscale region will be saved to the 'snips/' folder.")
    print(" - Press 'c' to cancel/clear current selection, 'q' to quit.")
    live_snip_tool(0)
