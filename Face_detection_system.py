
import cv2
import csv
from pathlib import Path
from datetime import datetime

# -------- CONFIGURATION --------

BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "known_faces"
ATTENDANCE_DIR = BASE_DIR / "attendance"
CSV_FILE = ATTENDANCE_DIR / "attendance.csv"

FACES_DIR.mkdir(exist_ok=True)
ATTENDANCE_DIR.mkdir(exist_ok=True)

# -------- FACE DETECTOR --------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -------- STUDENT REGISTER --------
# Example files:
# known_faces/101_Aman.jpg
# known_faces/102_Rahul.jpg

students = {}

for image_path in FACES_DIR.iterdir():
    if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        parts = image_path.stem.split("_", 1)

        if len(parts) == 2:
            student_id = parts[0]
            name = parts[1].replace("_", " ")
            students[student_id] = name

student_ids = sorted(students.keys())

# -------- CSV SETUP --------

if not CSV_FILE.exists():
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Student_ID", "Name", "Date", "Time"])

today = datetime.now().strftime("%Y-%m-%d")
marked_today = set()

with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row["Date"] == today:
            marked_today.add(row["Student_ID"])

def mark_attendance(student_id):
    if student_id in marked_today:
        print("Attendance already marked today.")
        return

    now = datetime.now()

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            student_id,
            students[student_id],
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ])

    marked_today.add(student_id)
    print("Attendance marked:", students[student_id])

# -------- WEBCAM --------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    exit()

print("Face Detection Attendance System")
print("Press student number key to mark attendance.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Cannot read webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

        # Show student list
        y_position = 60

        for index, student_id in enumerate(student_ids[:9]):
            text = f"{index + 1}: {students[student_id]}"
            cv2.putText(
                frame, text,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )
            y_position += 25

        cv2.imshow("Face Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF

        # Press number key to mark attendance
        if key in range(ord("1"), ord("9") + 1):
            index = key - ord("1")

            if index < len(student_ids) and len(faces) > 0:
                student_id = student_ids[index]
                mark_attendance(student_id)

        if key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()