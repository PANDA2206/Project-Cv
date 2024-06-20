import cv2
from mtcnn_cv2 import MTCNN
import numpy as np

class FaceDetector:
    def __init__(self, tm_window_size=60, tm_threshold=0.7, aligned_image_size=224):
        self.detector = MTCNN()
        self.reference = None
        self.aligned_image_size = aligned_image_size
        self.tm_window_size = tm_window_size
        self.tm_threshold = tm_threshold

    def detect_face(self, image):
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            # print("Detection empty")
            return None

        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]
        aligned = self.align_face(image, face_rect)
        self.reference = {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

        # # Draw a rectangle around the detected face for visualization
        # self.draw_rectangle(image, face_rect)
        # cv2.imshow('Face Detection', image)
        # cv2.waitKey(0)  # Add a small delay for visualization
        # # cv2.destroyAllWindows()

        return self.reference

    def track_face(self, image):
        if self.reference is None:
            return self.detect_face(image)

        ref_face = self.align_face(self.reference["image"], self.reference["rect"])
        # cv2.imshow("cropped", ref_face)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        search_window = [
            max(self.reference["rect"][0] - self.tm_window_size, 0),
            max(self.reference["rect"][1] - self.tm_window_size, 0),
            min(self.reference["rect"][0] + self.reference["rect"][2] + self.tm_window_size, image.shape[1]),
            min(self.reference["rect"][1] + self.reference["rect"][3] + self.tm_window_size, image.shape[0])
        ]

        search_region = image[search_window[0]:search_window[2],search_window[1]:search_window[3]]
        result = cv2.matchTemplate(search_region, ref_face, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # cv2.rectangle(image, (search_window[0], search_window[1]), (search_window[2], search_window[3]), (255, 0, 0), 2)
        # cv2.imshow('Face Detection', image)
        # cv2.waitKey()  # Add a small delay for visualization
        # cv2.destroyAllWindows()

        if max_val < self.tm_threshold:
            top_left = (max_loc[0] + search_window[0], max_loc[1] + search_window[1])
            bottom_right = (top_left[0] + self.reference["rect"][2], top_left[1] + self.reference["rect"][3])
            self.reference["rect"] = (*top_left, *bottom_right)
            self.reference["response"] = max_val

            # Draw a rectangle around the updated reference
            # Draw a rectangle around the search region for visualization
            # Draw a rectangle around the updated reference
            # self.draw_rectangle(image, self.reference["rect"])


        else:

            return None

        return self.reference

    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

    # def draw_rectangle(self, image, rect):
    #     cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

# # Create a FaceDetector instance
# face_detector = FaceDetector()
#
# # Open a connection to the camera (usually camera index 0)
# cap = cv2.VideoCapture(1)
#
# while True:
#     # Capture a frame from the camera
#     ret, frame = cap.read()
#
#     # Track faces in the captured frame
#     result = face_detector.track_face(frame)
#
#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the camera and close the OpenCV window
# cap.release()
# cv2.destroyAllWindows()

# face_detector = FaceDetector()
# img6 = cv2.imread("pank6.jpg")
# img7 = cv2.imread("pank7.jpg")
# img8 = cv2.imread("pank8.jpg")
# img = cv2.imread("0001.jpg")
#
# out = face_detector.track_face(img6)
# out2 = face_detector.track_face(img7)
# out3 = face_detector.track_face(img8)
# out4 = face_detector.track_face(img)
#cv2.imshow("out2",out2)

# ## Open a connection to the camera (usually camera index 0)
# cap = cv2.VideoCapture(0)
#
# while True:
#     # Capture a frame from the camera
#     ret, frame = cap.read()
#
#     # Track faces in the captured frame
#     result = face_detector.track_face(frame)
#
#     if result:
#         # Draw a rectangle around the detected/tracked face
#         rect = result["rect"]
#         cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
#
#     # Display the frame
#     cv2.imshow('Face Detection', frame)
#
#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(33) & 0xFF == ord('q'):
#         break

# Release the camera and close the OpenCV window
