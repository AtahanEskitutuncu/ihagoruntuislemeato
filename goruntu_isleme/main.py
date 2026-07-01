import cv2
from camera import Camera
from fps_counter import FPSCounter
from color_detection import ColorDetector
from logger import Logger


def main():
    logger = Logger()
    camera = Camera()
    fps_counter = FPSCounter()
    detector = ColorDetector(logger)

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        fps_counter.start()

        frame, mask, redMask, blueMask = detector.process(frame)

        instant_fps, avg_fps = fps_counter.stop()

        cv2.putText(
            frame,
            f"FPS: {int(instant_fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Average FPS: {int(avg_fps)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("kirmiziMaske", redMask)
        cv2.imshow("maviMaske", blueMask)

        

        if cv2.waitKey(3) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    