"""On-device text recognition with Apple Vision (macOS only, no network)."""
import cv2
import objc
import Vision
from Foundation import NSData


def read_lines(image_bgr, languages=('en-US',)):
    """Return recognised text lines as [(text, confidence, (x, y, w, h))] in image pixels."""
    with objc.autorelease_pool():
        ok, png = cv2.imencode('.png', image_bgr)
        if not ok:
            raise ValueError('cannot encode image for text recognition')
        data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        request.setRecognitionLanguages_(list(languages))
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f'Vision text recognition failed: {error}')
        h, w = image_bgr.shape[:2]
        lines = []
        for observation in request.results() or []:
            candidate = observation.topCandidates_(1)[0]
            box = observation.boundingBox()          # normalised, origin at the bottom left
            x, y = box.origin.x * w, (1 - box.origin.y - box.size.height) * h
            lines.append((str(candidate.string()), round(float(candidate.confidence()), 3),
                          (int(round(x)), int(round(y)), int(round(box.size.width * w)), int(round(box.size.height * h)))))
        return lines
