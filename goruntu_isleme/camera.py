import cv2

class Camera:
    def __init__(self, index=0, width=640, height=480, flip=True, blur_ksize=(5,5)):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        #cap = cv2.VideoCapture(0, cv2.CAP_V4L2) raspberry için
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.flip = flip
        self.blur_ksize = blur_ksize

    def read(self):
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None

        if self.flip:
            frame = cv2.flip(frame, 1)

        if self.blur_ksize is not None:
            frame = cv2.GaussianBlur(frame, self.blur_ksize, 0)

        return True, frame

    def release(self):
        self.cap.release()