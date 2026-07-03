import cv2

print("Trying to open camera...")
cap = cv2.VideoCapture(0)

print("Opened:", cap.isOpened())

if not cap.isOpened():
    print("PROBLEM: Camera did not open. Check if another app is using it,")
    print("or try VideoCapture(1) / VideoCapture(0, cv2.CAP_DSHOW) on Windows.")
else:
    ok, frame = cap.read()
    print("Frame read ok:", ok)
    if ok:
        print("Frame shape:", frame.shape)
        cv2.imshow("Test", frame)
        print("Press any key on the window to close...")
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

print("Done.")