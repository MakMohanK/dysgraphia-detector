import cv2
import os
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ------------------------
# Config
# ------------------------
OUTPUT_DIR = "snips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

drawing = False
ix = iy = -1
rx = ry = -1
rect_done = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load trained model
# ------------------------
model_path = r"C:\Users\Server\Documents\dysgraphia-detector\ml\Camera_ai\model\dysgraphia_model.pth"

# Define model structure (ResNet18 for 2 classes)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------------
# Define transforms & classes
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["dysgraphia", "normal"]

# ------------------------
# Prediction function
# ------------------------
def predict_image(image_path, model, transform, class_names):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    
    return class_names[pred.item()]

# ------------------------
# Mouse callback
# ------------------------
def mouse_callback(event, x, y, flags, param):
    global ix, iy, rx, ry, drawing, rect_done

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_done = False
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rx, ry = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry = x, y
        rect_done = True

        gray_frame = param['gray'].copy()
        h, w = gray_frame.shape
        x1, x2 = sorted((max(0, min(ix, w-1)), max(0, min(rx, w-1))))
        y1, y2 = sorted((max(0, min(iy, h-1)), max(0, min(ry, h-1))))

        if x2 - x1 > 5 and y2 - y1 > 5:
            crop = gray_frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(OUTPUT_DIR, f"snip_{timestamp}.png")
            cv2.imwrite(filename, crop)
            print(f"[Saved] {filename}")

            # ------------------------
            # Run prediction immediately
            # ------------------------
            pred_class = predict_image(filename, model, transform, class_names)
            print(f"[Prediction] {pred_class}")
            param['last_prediction'] = pred_class  # store for overlay
        else:
            print("[Info] Selection too small — not saved.")

# ------------------------
# Live snipping tool
# ------------------------
def live_snip_tool(camera_index=0):
    global ix, iy, rx, ry, drawing, rect_done

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("Grayscale (select to snip)")
    cv2.namedWindow("Original Feed (Rotated)")
    callback_param = {'gray': None, 'last_prediction': None}
    cv2.setMouseCallback("Grayscale (select to snip)", mouse_callback, callback_param)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: empty frame.")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold + invert
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 11)
        bw_inverted = cv2.bitwise_not(bw)
        callback_param['gray'] = bw_inverted

        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if drawing or rect_done:
            x1, y1 = ix, iy
            x2, y2 = rx, ry
            if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if callback_param.get('last_prediction'):
                    cv2.putText(disp, f"Prediction: {callback_param['last_prediction']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Original Feed (Rotated)", frame)
        cv2.imshow("Grayscale (select to snip)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            ix = iy = rx = ry = -1
            drawing = rect_done = False
            callback_param['last_prediction'] = None

    cap.release()
    cv2.destroyAllWindows()

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print("Instructions:")
    print(" - Click and drag on the 'Grayscale (select to snip)' window to select an area.")
    print(" - On mouse release, the snip is saved and automatically predicted.")
    print(" - Press 'c' to cancel selection, 'q' to quit.")
    live_snip_tool(0)
