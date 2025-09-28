import cv2

def live_text_black_white():
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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

        # Show both original and black/white output
        cv2.imshow("Original Feed", frame)
        cv2.imshow("Black & White Text", bw_inverted)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_text_black_white()
