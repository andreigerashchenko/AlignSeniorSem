import io

from kivy.core.window import Window
from kivy.graphics import Color, Ellipse
from kivy.uix.button import Button
from kivymd.toast import toast
from plyer import filechooser
from equirectRotate import EquirectRotate, pointRotate
import cv2
import numpy as np
import os
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.image import Image, AsyncImage
from kivy.core.image import Image as CoreImage
from kivymd.uix.toolbar import MDTopAppBar
from kivy.uix.carousel import Carousel
from kivymd.app import MDApp
from time import sleep
from kivy.config import Config
from kivy.lang import Builder
from queue import Queue
import moviepy

from horizonfinder import find_horizon, horizon_critical_points

Window.size = (1200, 750)
kv = Builder.load_file('main_screen.kv')


class HelpPopup(Popup):
    pass


class PrefPopup(Popup):
    pass


def getHorizonPoint(frame):
    pass


class DownloadPopup(Popup):
    pass


class HistoryItem():
    def __init__(self, img, flipV, flipH):
        self.img = img
        self.flipV = flipV
        self.flipH = flipH


class MainScreen(BoxLayout):
    popup = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__()
        self.modifyHistory = True
        self.touchLocalX = None
        self.touchLocalY = None
        self.selectedPoint = None
        self.fileQueue = []  # the list of files queued to be processed
        self.mediaPath = None  # the path to the file currently being worked on

        # stack history of transformations applied to the image,
        # history is reset when a new file is selected
        self.history = []

        # openCV image array of the current image being previewed, previous  version of  this
        self.currentImg = None

        self.previewimgPath = os.path.abspath(".previewImg.jpg")
        print(os.listdir(os.getcwd()))
        startImg = cv2.imread("no_img.png")

        cv2.imwrite(self.previewimgPath, startImg, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def openFileBrowser(self):
        # save original directory to restore at end
        cwd = os.getcwd()

        file_path = ""
        file_path = filechooser.open_file(title="File Selection", multiple=True, filters=[
            "*.jpg", "*.png", "*.jpeg", "*.mp4"])
        if file_path == []:
            pass
        else:
            self.fileQueue.extend(file_path)
            self.mediaPath = self.fileQueue[0]
            self.currentImg = cv2.imread(self.mediaPath)
            self.updateImage(self.currentImg)

            # load thumbnails to the queue carousel object
            queueThumbnails = self.ids.imgQueue
            for file in file_path:
                filename = os.path.split(file)[-1]

                # create image button of the selected file
                im = Button(background_normal=file, size_hint=(None, 1), width=100,
                            text=filename, font_size=10, on_press=lambda image: self.focusImage(image))

                # add the buttonImage to the queue
                queueThumbnails.add_widget(im)

                print("added")

        # restore original directory
        os.chdir(cwd)

    def focusImage(self, img):
        newImg = cv2.imread(img.background_normal)
        self.updateImage(newImg)

        # clear history
        self.history.clear()

    def open_Help(self):
        self.popup = HelpPopup()
        self.popup.open()

    def open_popup(self):
        self.popup = PrefPopup()
        self.popup.open()

    # pops up a menu for the user to select a file name and save image to that file
    def saveImagePopup(self):
        self.popup = DownloadPopup()
        self.popup.open()

    # uses the current preview image and the file name from the popup to save a new image file
    def saveImage(self):
        popup = self.get_root_window().children[0]
        outfile = popup.ids.saveFileNameInput.text
        cv2.imwrite(outfile, self.currentImg, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 100])
        toast(f"saved as {outfile}")

    # replace with the function which does some calculation to maintain progressbar value

    def press_it(self):
        print(self.ids.mirrorX_switch.active)
        # Grab the current progress bar value
        current = self.ids.my_progress_bar.value
        current2 = self.ids.my_progress_bar.value
        # Increment value by .25
        current += 25

        current2 += 29
        # If statement to start over after 100
        if current > 100:
            current = 0
            current2 = 0
        # Update the progress bar
        self.ids.my_progress_bar.value = current
        sleep(0.31)
        self.ids.my_progress_bar2.value = current2
        # Update the label
        # self.ids.my_label.text = f'{int(current)}% Progress'

    # see doc MDProgress bar

    def press_it2(self):
        # Grab the current progress bar value
        current = self.progression_value
        # If statement to start over after 100
        if current == 100:
            current = 0

        # Increment value by .25
        current += 25

        # Update the label
        # self.ids.my_label2.text = f'{int(current)}% Progress'

    """
    What happens when you click on the window (sepcificallly on the image)
    saves the local coordinates of the user's click on the image.
    draws a circle around where the user clicked to inform user of click location 
    Return: None
    """

    def on_touch_up(self, touch):

        # touch provides the click's global cords for the entire app
        # localize the touch coordinates to the image
        previewImg = self.ids.previewImage
        imgCords = previewImg.pos
        imgSize = previewImg.size

        # undo width stretch
        # assumes image width is always 2x image height
        stretchWidth = imgSize[0]
        imgSize[0] = imgSize[1] * 2
        stretchGap = (stretchWidth - imgSize[0])

        # get coordinates of mouse click relative to bottom left of image
        touchLocalX = touch.x - imgCords[0] - stretchGap / 2
        touchLocalY = touch.y - imgCords[1]

        # if click is outside of image bounds, disregard it
        if touchLocalX < 0 or touchLocalX > imgSize[0]:
            return
        if touchLocalY < 0 or touchLocalY > imgSize[1]:
            return

        self.touchLocalX = touchLocalX
        self.touchLocalY = touchLocalY

        # if there is no image to be worked on, open file browser
        if not self.mediaPath:
            self.openFileBrowser()
            return

        # draw cricle to represent selected area
        with self.canvas:
            # remove selected point from image
            if self.selectedPoint:
                self.canvas.remove(self.selectedPoint)
                self.selectedPoint = None

            Color(.75, .3, .3, .6)
            d = 15.
            self.selectedPoint = Ellipse(
                pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))

    def processImage(self):

        manual = self.ids.manual_switch.active

        if manual:
            self.manualProcess()
        else:
            self.automaticProcess()

    def automaticProcess(self):
        img = self.currentImg
        scale_factor = min(1280 / img.shape[1], 720 / img.shape[0])
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

        horizon_contour = find_horizon(img)
        # Draw the contour on the image
        cv2.drawContours(img, [horizon_contour], -1, (0, 255, 0), 2)

        critical_points = horizon_critical_points(img,horizon_contour)
        cx = int(critical_points[0][0])
        cy = int(critical_points[0][1])

        cv2.circle(img,(cx,cy), 10, (0,0,255), -1)

        self.updateImage(img)


    """
    Uses the point selected  by the user to equirotate the preview Image
    Return: None
    """

    def manualProcess(self):
        # if no point selected, message the user and return
        if not self.selectedPoint:
            toast("no point selected")
            return

        # remove selected point from image
        self.canvas.remove(self.selectedPoint)
        self.selectedPoint = None

        previewImg = self.ids.previewImage
        imgSize = previewImg.size

        # check for mirror X and mirror Y settings
        mirrorX = self.ids.mirrorX_switch.active
        mirrorY = self.ids.mirrorY_switch.active

        # # flip Y cordinate if mirrorY is active
        # if not mirrorY:
        self.touchLocalY = -(self.touchLocalY - imgSize[1])

        # get image paths for input and output
        src_path = previewImg.source
        opfile = self.previewimgPath

        # open the image to be transformed
        src_image = cv2.imread(previewImg.source)

        # scale touch coordinates to image size
        h, w, c, ix, iy = scaleImage(
            src_image, imgSize, self.touchLocalX, self.touchLocalY)
        print(f"Clicked Location (x,y): {ix},{iy}")

        # rotate the image and update the preview
        rotatedImage = rotateImage(
            src_image, h, w, c, ix, iy, mirrorX, mirrorY)

        self.updateImage(rotatedImage)

    # flips the current image vertically
    def flipVertical(self):
        flippedImage = cv2.flip(self.currentImg, 0)
        self.updateImage(flippedImage, flipV_inverse=True)

    # flips the current image horizontally
    def flipHorizontal(self):
        flippedImage = cv2.flip(self.currentImg, 1)
        self.updateImage(flippedImage, flipH_inverse=True)

    """
    undoes the last change the user made the image
    gets the last frame from the history and restores 
    image and settings to that of last application frame
    does nothing if the history is empty
    history is reset once the image path is changed
    
    Return: None
    """

    def undo(self):
        # if history is empty, notify user adn do nothing
        if len(self.history) == 0:
            toast("history is empty")
            return

        self.modifyHistory = False

        # get the last state of the history
        lastState = self.history.pop()

        print(lastState.flipV, lastState.flipH)

        # set the switches to reflect the states at that point in history
        self.ids.mirrorY_switch.active = lastState.flipV
        self.ids.mirrorX_switch.active = lastState.flipH

        # change the preview image to that of the history frame
        self.updateImage(lastState.img)

        self.modifyHistory = True

    """
    Takes an image and updates the application preview to display it
    stores the current state of the app to history
    Params:
    newImg - openCV image array of the image to update the preview to
    flipV_inverse - whether to invert the state of the vertical flip switch
    flipH_inverse - whether to invert the state of the Horizontal flip switch
    
    Return: None
    """

    def updateImage(self, newImg, flipV_inverse=False, flipH_inverse=False):

        # if modifyHistory, store the current Image in history
        if self.modifyHistory:
            flipV = self.ids.mirrorY_switch.active
            if flipV_inverse:
                flipV = not flipV

            flipH = self.ids.mirrorX_switch.active
            if flipH_inverse:
                flipH = not flipH

            histItem = HistoryItem(self.currentImg, flipV, flipH)

            self.history.append(histItem)

        # set the current image to the new one
        self.currentImg = newImg

        # save the previewImage and update the visual
        cv2.imwrite(self.previewimgPath, newImg, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 100])
        self.ids.previewImage.reload()

    '''
    processFrame
    takes a bitmaap array and video timing data to process into it's equirotated state
    frame: a bitmap array of a frame of a video
    frameNum: the number of the specific frame out of the video
    interval: the interval in which the horizon location will be recalculated after [interval] frames
    '''
    # def processFrame(self,frame,frameNum,interval,rotator):
    #     frameIntervalProgress = frameNum % interval
    #
    #     # when at frame interval
    #     # find the horizon of the current feame
    #     if frameIntervalProgress == 0:
    #         horizonX,horizonY = getHorizonPoint(frame)
    #
    #     equiRotateFrame(frame,horizonX,horizonY)
    #


"""
    scales the local coordinates on an image to the size of another image
    
    Params:
    src_image - the target image to scale the coordinates to
    imgSize - the tuple (h,w) size of the input image
    localX,localY - the local coordinates of the input image
    
    Return: 
    h,w,c - dimensions of the output image
    ix,iy - the local coordinates scaled to the output image
"""


def scaleImage(src_image, imgSize, localX, localY):
    h, w, c = src_image.shape

    ratio = w / imgSize[0]

    ix = localX * ratio
    iy = localY * ratio

    return h, w, c, ix, iy


def rotateImage(src_image, h, w, c, ix, iy, mirrorX, mirrorY):
    # everything after this is dipankr's code
    ###################################################################
    print('\n Now rotating the image to straighten the horizon.')
    print("\n Input file's height, width, colors =", h, w, c)

    # Do a 'yaw' rotation such that ix position earth-sky horizon is
    # at the middle column of the image. Fortunately for an equirectangular
    # image, a yaw is simply sliding the image horizontally, and is done very
    # fast by np.roll.
    shiftx = int(w / 2 - ix)
    src_image = np.roll(src_image, shiftx, axis=1)

    # If iy>0 then the user selected the lowest point of the horizon.
    # After the above 'yaw', the true horizon at the middle of the image
    # is still (iy - h/2) pixels below the camera's equator. This is
    # (iy - h/2)*(180)/h degrees below the camera's equator. So rotate the
    # pitch of the yaw-ed rectilinear image by this amount to get a nearly
    # straight horizon.
    myY, myP, myR = 0, (iy - h / 2) * 180 / h, 0

    # If iy<0 then the user actually recorded the highest point. That
    # is, the true horizon is (h/2 - |iy|) pixels above the camera's
    # equator. So rotate the pitch of the yaw-ed rectilinear image by the
    # amount -(h/2 - |iy|)*180/h to get a nearly straight horizon.
    if iy < 0:
        myP = -(h / 2 - np.abs(iy)) * 180 / h

    print('\n Doing the final rotation (pitch =', str(f'{myP:.2f}'),
          'deg). This can take a while ...')
    # rotate (yaw, pitch, roll)
    equirectRot = EquirectRotate(h, w, (myY, myP, myR))
    rotated_image = equirectRot.rotate(src_image)
    ###################################################################

    # # mirror axis WIP
    # if mirrorX and not mirrorY:
    #     flipAxis = 1
    # if mirrorY and not mirrorX:
    #     flipAxis = 0
    # if mirrorX and mirrorY:
    #     flipAxis = -1
    #
    # if not (mirrorX or mirrorY):
    #     final_image = rotated_image
    # else:
    #     final_image = cv2.flip(rotated_image, flipAxis)

    return rotated_image


class MainScreenApp(MDApp):

    def build(self):
        return MainScreen()


MainScreenApp().run()
