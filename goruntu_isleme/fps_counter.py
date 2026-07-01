import time

class FPSCounter:
    def __init__(self):
        self.start_time = 0
        self.instant_fps = 0
        self.avg_fps = 0
        self.count = 0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        dt = time.perf_counter() - self.start_time

        if dt > 0:
            self.instant_fps = 1.0 / dt

        self.count += 1

        if self.count == 1:
            self.avg_fps = self.instant_fps
        else:
            self.avg_fps = (
                self.avg_fps * (self.count - 1) + self.instant_fps
            ) / self.count

        return self.instant_fps, self.avg_fps