
import logging

class Logger:
    def __init__(self, log_file="vision.log"):
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ] 
            )

        self.logger = logging.getLogger("VisionLogger")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)