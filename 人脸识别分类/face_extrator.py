from openface import align_dlib
import cv2
from matplotlib import pyplot

class FaceExtractor:
    def __init__(self):
        self.align = align_dlib.AlignDlib("face_model/shape_predictor_68_face_landmarks.dat")

    def extrat_face_img(self, input_img_path, output_size=96):
        img = cv2.imread(input_img_path)
        bb = self.align.getLargestFaceBoundingBox(img)
        if not bb:
            return None, None, None
        face_img = self.align.align(100, img, bb=bb, landmarkIndices=align_dlib.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        landmarkers = self.align.findLandmarks(img, bb)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return face_img, bb, landmarkers

    def vis_face_img(self, input_img_path, bb, landmarkers):
        img = cv2.imread(input_img_path)
        cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0, 255, 0), thickness=3)
        for pt in landmarkers:
            cv2.circle(img, pt, 2, (0, 255, 255))
        cv2.imshow("face", img)
        cv2.waitKey()

    def vis_face_imgs(self, images):
        pyplot.figure()
        col = 4
        row = int(len(images) / col) + 1
        for i in range(0, len(images)):
            pyplot.subplot(row, col, i + 1)
            pyplot.imshow(images[i])
        pyplot.show()
