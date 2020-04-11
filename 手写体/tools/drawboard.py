import cv2
import numpy as np


class DrawBoard:
    current_line: [(float, float)]
    is_drawing: bool

    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.cvCanvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.current_line = []
        self.is_drawing = False
        self.user_draw_cb = None

    def onmouse(self, event, x, y, flags, params):
        rect = np.array([[[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]], dtype=np.int32)
        cv2.fillPoly(self.cvCanvas, rect, (255, 255, 255))
        need_trigger_cb = False
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.current_line.clear()
            self.current_line.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.current_line.append((x, y))
        else:
            self.is_drawing = False
            need_trigger_cb = True

        if len(self.current_line) > 1:
            for i in range(1, len(self.current_line) - 1):
                cv2.line(self.cvCanvas, self.current_line[i-1], self.current_line[i], (0, 0, 0), 25)

        if need_trigger_cb:
            if self.user_draw_cb:
                self.user_draw_cb(self.cvCanvas)

        cv2.imshow("board", self.cvCanvas)

    def run(self, draw_cb):
        self.user_draw_cb = draw_cb
        cv2.imshow("board", self.cvCanvas)
        cv2.setMouseCallback("board", lambda event, x, y, flags, params: self.onmouse(event, x, y, flags, params))
        cv2.waitKey()


if __name__ == '__main__':
    draw_board = DrawBoard(280, 280)
    def user_draw_cb(img):
        print(img.shape)
    draw_board.run(user_draw_cb)
