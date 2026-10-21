import cv2
import numpy as np

class OCR:
    def __init__(self):
        import torch
        from paddleocr import PaddleOCR
        import cv2
        import numpy as np

        self.ocr = PaddleOCR(use_angle_cls=True, lang="latin")  # Set 'lang' as needed

    def detect(self, image):
        image = self.preprocess(image)
        result = self.ocr.ocr(image, cls=True)
        return result[0][0][-1][0]
    
    def preprocess(self, image):
        plate = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(2.0, (8,8))
        gray = clahe.apply(gray)

        gray = cv2.GaussianBlur(gray, (3,3), 0)

        return gray
    
class FastALPROCR:
    def __init__(self, end_to_end=False):
        # from fast_alpr import ALPR
        from fast_plate_ocr import LicensePlateRecognizer

        self.mode = end_to_end
        if end_to_end:
            self.model = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-xs-v2-global-model",
            )

        else:
            self.model = LicensePlateRecognizer('cct-xs-v2-global-model')


    def detect(self, image: np.ndarray) -> str:
        """
        Predict the license plate number of a license plate image
        """
        if self.mode:
            result = self.model.predict(image)

            return result[0].ocr.text

        else:
            result = self.model.run_one(image)

            return result.plate

if __name__ == "__main__":
    ocr = FastALPROCR(end_to_end=True)
    # ocr = OCR()
    img = cv2.imread(r"C:\Users\AQI\Downloads\vlcsnap-2026-06-18-11h32m25s585.png")

    result = ocr.detect(img)
    print(result)
