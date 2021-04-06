import sys,os,json,collections
import tifffile as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import rotate
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, uic

def rotateAndScale(img, scaleFactor = 0.5, InPlaneRot_Degree = 30):
    (oldY,oldX) = np.shape(img) #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=InPlaneRot_Degree, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(InPlaneRot_Degree)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M2 = M
    M2[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M2[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M2, dsize=(int(newX),int(newY)))
    return M2,rotatedImg

def rotateScaleTranslate(img, Translation=(200, 500), scaleFactor=0.5, InPlaneRot_Degree=30):
    (oldY, oldX) = np.shape(img)  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    print(oldX, oldY)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=InPlaneRot_Degree,
                                scale=scaleFactor)  # rotate about center of image.
    print(M)

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(InPlaneRot_Degree)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.

    M[0, 2] += Translation[0]  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += Translation[1]

    rotatedImg = cv2.warpAffine(np.float32(img), M, (int(newX), int(newY)))
    return M, rotatedImg

def rotate_box(bb, cx, cy, h, w, theta=5):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        theta = np.radians(theta)
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        #new_bb[i] = (calculated[0],calculated[1])
    return calculated[0],calculated[1]

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


class ImageCorrelationWindow(QtWidgets.QMainWindow):
    def __init__(self, ref_image=None):
        super(ImageCorrelationWindow, self).__init__()
        uic.loadUi('imageCorrelation.ui', self)
        self.ref_image = ref_image
        self.coords = collections.deque(maxlen=4)

        # connections
        self.actionLoad_refImage.triggered.connect(self.loadRefImage)
        self.pb_apply_calculation.clicked.connect(self.scalingCalculation)
        self.dsb_x_off.valueChanged.connect(self.offsetCorrectedPos)
        self.dsb_y_off.valueChanged.connect(self.offsetCorrectedPos)
        self.pb_grabXY_1.clicked.connect(self.insertCurrentPos1)
        self.pb_grabXY_2.clicked.connect(self.insertCurrentPos2)
        self.pb_import_param.clicked.connect(self.importScalingParamFile)
        self.pb_export_param.clicked.connect(self.exportScalingParamFile)
        self.pb_gotoTargetPos.clicked.connect(self.gotoTargetPos)

    def loadRefImage(self):
        self.file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Select Ref Image", '',
                                                                 'image file(*png *jpeg *tiff *tif )')
        if self.file_name:
            self.ref_image = plt.imread(self.file_name[0])
            if self.ref_image.ndim == 3:
                self.ref_image = self.ref_image.sum(2)
            self.statusbar.showMessage(f'{self.file_name[0]} selected')
            self.yshape, self.xshape = np.shape(self.ref_image)
        else:
            self.statusbar.showMessage("No file has selected")
            pass

        try:
            self.ref_view.clear()
        except:
            pass


        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.ref_view.addPlot(title="")

        # Item for displaying image data
        self.img = pg.ImageItem()
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img)
        self.ref_view.addItem(hist)

        self.p1.addItem(self.img)
        self.ref_image = rotate(self.ref_image, -90)
        self.img.setImage(self.ref_image)
        self.img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        # self.img.translate(100, 50)
        # self.img.scale(0.5, 0.5)
        self.img.hoverEvent = self.imageHoverEvent
        self.img.mousePressEvent = self.MouseClickEvent

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
        j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
        val = self.ref_image[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = np.around(ppos.x(), 2), np.around(ppos.y(), 2)
        self.p1.setTitle(f'pos: {x, y}  pixel: {i, j}  value: {val}')

    def MouseClickEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """

        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            i, j = pos.x(), pos.y()
            i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
            j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))

            if self.rb_calib_mode.isChecked():
                self.coords.append((i, j))
                val = self.ref_image[i, j]
                ppos = self.img.mapToParent(pos)
                x, y = np.around(ppos.x(), 2) , np.around(ppos.y(), 2) #mm to um
                # x, y = smarx.pos, smary.pos
                self.coords.append((x, y))
                if len(self.coords) == 2:
                        self.le_ref1_pxls.setText(f'{self.coords[0][0]}, {self.coords[0][1]}')
                        self.dsb_ref1_x.setValue(self.coords[1][0])
                        self.dsb_ref1_y.setValue(self.coords[1][1])
                elif len(self.coords) == 4:
                        self.le_ref1_pxls.setText(f'{self.coords[0][0]},{self.coords[0][1]}')
                        self.dsb_ref1_x.setValue(self.coords[1][0])
                        self.dsb_ref1_y.setValue(self.coords[1][1])
                        self.le_ref2_pxls.setText(f'{self.coords[2][0]},{self.coords[2][1]}')
                        self.dsb_ref2_x.setValue(self.coords[-1][0])
                        self.dsb_ref2_y.setValue(self.coords[-1][1])

            elif self.rb_nav_mode.isChecked():
                (h, w) = self.ref_image.shape[:2]
                (cx, cy) = (w // 2, h // 2)
                (new_h, new_w) = self.affineImage.shape[:2]
                (new_cx, new_cy) = (new_w // 2, new_h // 2)
                angle = np.radians(self.dsb_rotAngle.value())

                self.affineMatrix, self.affineImage = rotateAndScale(self.ref_image, scaleFactor=self.pixel_val_x,
                                                                     InPlaneRot_Degree=self.dsb_rotAngle.value())

                #self.xWhere, self.yWhere = rotate_box([[i,j]],cx,cy,h,w,self.dsb_rotAngle.value())
                '''
                self.xWhere = (i-cx)*np.cos(angle)-((j-cy)*np.sin(angle)) +cx
                self.yWhere = (j-cy)*np.cos(angle)+((i-cx)*np.sin(angle)) +cy
                
                print(self.affineMatrix[0, 2],self.affineMatrix[0, 2])
                cos = self.affineMatrix[0,0]
                sin = self.affineMatrix[0,1]
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                xDiff, yDiff = nW/2 - cx, nH/2 - cy
                self.affineMatrix[0, 2] += xDiff
                self.affineMatrix[1, 2] += yDiff
                #xDiff, yDiff = ((1-al)*cx - bt*cy)+((new_w/2)-cx),((1-al)*cy + bt*cx)+((new_h/2)-cy)
                '''

                self.xWhere = (i-cx)*np.cos(angle)-((j-cy)*np.sin(angle)) +cx + np.cos(angle)*((new_cx-cx)/2)
                self.yWhere = (j-cy)*np.cos(angle)+((i-cx)*np.sin(angle)) +cy + (new_h-h)/2

                #self.xWhere, self.yWhere = self.affineMatrix @ [i, j, 1]
                self.rectROI.setPos((self.xWhere, self.yWhere), y = None, update = True, finish = True)
                print(f'oldShape: {(h,w)} , NewShape: {np.shape(self.affineImage)}')
                print(f'Ref pixels{i, j}')
                print(f'xWhere;{self.xWhere:.1f},yWhere;{self.yWhere:.1f}')
                print(f' ROI_Pos = {self.rectROI.pos()}')
                self.offsetCorrectedPos()

    def createLabAxisImage(self, image):
        # A plot area (ViewBox + axes) for displaying the image

        try:
            self.labaxis_view.clear()
        except:
            pass


        self.p2 = self.labaxis_view.addPlot(title="")

        # Item for displaying image data
        self.img2 = pg.ImageItem()
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img2)
        self.labaxis_view.addItem(hist)
        self.p2.addItem(self.img2)
        self.img2.setImage(image)
        imX,imY = image.shape[:2]
        self.rectROI = pg.RectROI([int(imX // 2), int(imX // 2)],
                                  [imY//10, imY//10], pen='r')
        self.p2.addItem(self.rectROI)
        #self.img2.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        #self.img2.setImage(self.ref_image.T,opacity = 0.5)

    def getScalingParams(self):

        self.lm1_px, self.lm1_py = self.le_ref1_pxls.text().split(',')  # r chooses this pixel
        self.lm2_px, self.lm2_py = self.le_ref2_pxls.text().split(',')  # chooses this pixel

        # motor values from the microscope at pixel pos 1
        self.lm1_x, self.lm1_y = self.dsb_ref1_x.value(), self.dsb_ref1_y.value()
        # motor values from the microscope at pixel pos 2
        self.lm2_x, self.lm2_y = self.dsb_ref2_x.value(), self.dsb_ref2_y.value()
        self.rb_calib_mode.setChecked(False)
        self.rb_nav_mode.setChecked(True)

    def exportScalingParamFile(self):
        self.getScalingParams()
        self.scalingParam = {}
        ref_pos1 = {'px1': int(self.lm1_px), 'py1':int(self.lm1_py), 'cx1':self.lm1_x, 'cy1':self.lm1_y}
        ref_pos2 = {'px2': int(self.lm2_px), 'py2': int(self.lm2_py), 'cx2': self.lm2_x, 'cy2': self.lm2_y}
        self.scalingParam['lm1_vals'] = ref_pos1
        self.scalingParam['lm2_vals'] = ref_pos2

        file_name = QtWidgets.QFileDialog().getSaveFileName(self, "Save Parameter File", 'scaling_parameters.json',
                                                                 'json file(*json)')
        if file_name:

            with open(f'{file_name[0]}', 'w') as fp:
                json.dump(self.scalingParam,fp, indent=4)
        else:
            pass

    def importScalingParamFile(self):
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Open Parameter File", '',
                                                                 'json file(*json)')
        if file_name:
            with open(file_name[0], 'r') as fp:
                self.scalingParam = json.load(fp)
        else:
            pass

        px1, py1 = self.scalingParam['lm1_vals']['px1'], self.scalingParam['lm1_vals']['py1']
        px2, py2 = self.scalingParam['lm2_vals']['px2'], self.scalingParam['lm2_vals']['py2']

        self.le_ref1_pxls.setText(f'{px1},{py1}')
        self.dsb_ref1_x.setValue(self.scalingParam['lm1_vals']['cx1'])
        self.dsb_ref1_y.setValue(self.scalingParam['lm1_vals']['cy1'])
        self.le_ref2_pxls.setText(f'{px2},{py2}')
        self.dsb_ref2_x.setValue(self.scalingParam['lm2_vals']['cx2'])
        self.dsb_ref2_y.setValue(self.scalingParam['lm2_vals']['cy2'])

    def scalingCalculation(self):
        self.getScalingParams()
        # pixel value of X
        self.pixel_val_x = (self.lm2_x - self.lm1_x) / (int(self.lm2_px) - int(self.lm1_px))
        # pixel value of Y; ususally same as X
        self.pixel_val_y = (self.lm2_y - self.lm1_y) / (int(self.lm2_py) - int(self.lm1_py))

        self.xi = self.lm1_x - (self.pixel_val_x * int(self.lm1_px))  # xmotor pos at origin (0,0)
        xf = self.xi + (self.pixel_val_x * self.xshape)  # xmotor pos at the end (0,0)
        self.yi = self.lm1_y - (self.pixel_val_y * int(self.lm1_py))  # xmotor pos at origin (0,0)
        yf = self.yi + (self.pixel_val_y * self.yshape)  # xmotor pos at origin (0,0)

        self.affineMatrix, self.affineImage = rotateAndScale(self.ref_image, scaleFactor = self.pixel_val_x,
                                                             InPlaneRot_Degree = self.dsb_rotAngle.value())
        #self.affineMatrix, self.affineImage = rotateScaleTranslate(self.ref_image, Translation = (0,0), scaleFactor = self.pixel_val_x,
                                                             #InPlaneRot_Degree = self.dsb_rotAngle.value())

        #self.affineImage = rotate_bound(self.ref_image,self.dsb_rotAngle.value())

        self.createLabAxisImage(self.affineImage)

        self.label_scale_info.setText(f'Scaling: {self.pixel_val_x:.4f}, {self.pixel_val_y:.4f}, \n '
                                      f' X Range {self.xi:.2f}:{xf:.2f}, \n'
                                      f'Y Range {self.yi:.2f}:{yf:.2f}')
        #self.img2.scale(abs(self.pixel_val_x), abs(self.pixel_val_y))
        #self.img2.translate(self.xi, self.yi)
        # self.img2.setRect(QtCore.QRect(xi,yf,yi,xf))
        #self.img2.hoverEvent = self.imageHoverEvent2
        self.img2.mousePressEvent = self.MouseClickEventToPos

    def imageHoverEvent2(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p2.setTitle("")
            return
        pos = event.pos()
        i, j = pos.x(), pos.y()
        i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
        j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
        val = self.ref_image[i, j]
        x = self.xi + (i)
        y = self.yi + (j)
        self.p2.setTitle(f'pos: {x:.2f},{y:.2f}  pixel: {i, j}  value: {val:.2f}')

    def MouseClickEventToPos(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            i, j = pos.x(), pos.y()
            i = int(np.clip(i, 0, self.ref_image.shape[0] - 1))
            j = int(np.clip(j, 0, self.ref_image.shape[1] - 1))
            self.xWhere = self.xi + i
            self.yWhere = self.yi + j
            self.offsetCorrectedPos()

    def offsetCorrectedPos(self):
        self.dsb_calc_x.setValue(self.xWhere + (self.dsb_x_off.value() * 0.001))
        self.dsb_calc_y.setValue(self.yWhere + (self.dsb_y_off.value() * 0.001))

    def insertCurrentPos1(self):
        try:
            posX = smarx.position
            posY = smary.position
        except:
            posX = 0
            posY = 0

        self.dsb_ref1_x.setValue(posX)
        self.dsb_ref1_y.setValue(posY)

    def insertCurrentPos2(self):
        try:
            posX = smarx.position
            posY = smary.position
        except:
            posX = 1
            posY = 1

        self.dsb_ref2_x.setValue(posX)
        self.dsb_ref2_y.setValue(posY)

    def gotoTargetPos(self):
        targetX = self.dsb_calc_x.value()
        targetY = self.dsb_calc_y.value()
        try:
            RE(bps.mov(smarx, targetX))
            RE(bps.mov(smary, targetY))
        except:
            print (targetX,targetY)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = ImageCorrelationWindow()
    window.show()
    sys.exit(app.exec_())
