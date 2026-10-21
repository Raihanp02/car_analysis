from ocr import OCR

class LicensePlatePipelines:
    def __init__(self, plate_detector, region: list, ocr):
        self.plate_detector = plate_detector
        self.ocr = ocr

    def process(self, image, inside_region):
        result = self.plate_detector.detect(image)

        for bbox in result:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            if x1 >= self.region[0] and y1 >= self.region[1] and x2 <= self.region[2] and y2 <= self.region[3]:
                parsed = self.ocr.ocr(image[y1:y2, x1:x2])

    
