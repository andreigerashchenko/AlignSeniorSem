import io
import shutil

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse
from kivy.uix.button import Button
from kivymd.toast import toast
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import *
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
from kivy.uix.videoplayer import VideoPlayer
from queue import Queue
import moviepy

pbcurrent = 0
from horizonfinder import find_horizon_point

Window.size = (1200, 750)
kv = Builder.load_file('main_screen.kv')

vidPreviewPath = os.path.abspath(".vidPreview.mp4")
previewimgPath = os.path.abspath(".previewImg.jpg")

class HelpPopup(Popup):
    pass


class PrefPopup(Popup):
    pass


class VidPreviewPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__()
        self.vidPreviewPath = os.path.abspath(".vidPreview.mp4")
    def browseFile(self):
        file_select = filechooser.open_file(title="Save Video")
        if not file_select:
            return
        filename = file_select[0]
        self.ids.saveFileNameInput.text = filename

    def saveVideo(self):
        outfile_name = self.ids.saveFileNameInput.text

        # do not accept empty path
        if outfile_name == "":
            toast("no file path")
            return

        # check if the file path exists
        file_dir = os.path.dirname(outfile_name)
        if not os.path.exists(file_dir):
            toast("invalid file path")
            return

        # add mp4 extension if not present
        if not outfile_name.lower().endswith(".mp4"):
            outfile_name += ".mp4"

        # save preview video to outfile
        shutil.copy(vidPreviewPath,outfile_name)
        toast(f"saved as {outfile_name}")
        self.dismiss()

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
    global pbcurrent
    popup = ObjectProperty(None)
    if pbcurrent > 100:
        pbcurrent = 0

    def __init__(self, **kwargs):
        super().__init__()
        self.image = 0
        self.video = 1
        self.currentMediaType = None
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
        self.vidPreviewPath = os.path.abspath(".vidPreview.mp4")

        print(os.listdir(os.getcwd()))
        startImg = cv2.imread("no_img.png")

        cv2.imwrite(self.previewimgPath, startImg, [
            int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print("Wrote no_img.png to .previewImg.jpg")

    def openFileBrowser(self):
        # save original directory to restore at end
        cwd = os.getcwd()

        file_path = ""
        file_path = filechooser.open_file(title="File Selection", multiple=True, filters=[
            "*.jpg", "*.png", "*.jpeg", "*.mp4"])
        if file_path is None or file_path == []:
            pass
        else:
            self.fileQueue.extend(file_path)
            self.focusMedia(self.fileQueue[0])
            # self.mediaPath = self.fileQueue[0]
            # self.currentImg = cv2.imread(self.mediaPath)
            # self.updateImage(self.currentImg)

            # load thumbnails to the queue carousel object
            queueThumbnails = self.ids.imgQueue
            for file in file_path:
                filename = os.path.split(file)[-1]
                extenstion = os.path.splitext(filename)[-1][1:]

                # if video, create video button of selected file
                if extenstion in ["mp4", "mov"]:
                    im = Button(background_normal="blank_video_logo.png", size_hint=(None, 1), width=100,
                                text=filename, font_size=10, on_press=lambda vid: self.focusVideo(file))

                # if image, create image button of the selected file
                else:
                    im = Button(background_normal=file, size_hint=(None, 1), width=100,
                                text=filename, font_size=10, on_press=lambda image: self.focusImage(file))

                # add the buttonImage to the queue
                queueThumbnails.add_widget(im)

        # restore original directory
        os.chdir(cwd)

    def focusMedia(self, mediaPath):
        self.mediaPath = mediaPath
        extenstion = os.path.splitext(mediaPath)[-1][1:]

        # if video, create video button of selected file
        if extenstion in ["mp4", "mov"]:
            self.focusVideo(mediaPath)
        else:
            self.focusImage(mediaPath)

    def focusImage(self, img):
        newImg = cv2.imread(img)
        self.updateImage(newImg)

        # clear history
        self.history.clear()

        # set current media to image
        self.mediaPath = img
        self.currentMediaType = self.image

        # enable the manual switch
        self.ids.manual_switch.disabled = False

    def focusVideo(self, vidPath):
        # get the video from path
        clip = VideoFileClip(vidPath)

        # getting only first 5 seconds
        clip = clip.subclip(0, 5)

        # getting frame at time 0
        frame = clip.get_frame(0)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = drawVideoLogo(frame)

        self.updateImage(frame)

        # clear history
        self.history.clear()

        # set current media to video
        self.mediaPath = vidPath
        self.currentMediaType = self.video

        # disable the manual switch, only automatic can be used for videos
        self.ids.manual_switch.active = False
        self.ids.manual_switch.disabled = True

    def open_Help(self):
        self.popup = HelpPopup()
        self.popup.open()

    def vidPreview(self):
        self.popup = VidPreviewPopup()
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
        outfile = file_path = filechooser.open_file(title="Save Image")
        cv2.imwrite(outfile, self.currentImg, [
            int(cv2.IMWRITE_JPEG_QUALITY), 100])
        toast(f"saved as {outfile}")

    # replace with the function which does some calculation to maintain progressbar value

    def press_it(self):

        # Grab the current progress bar value

        current2 = self.ids.my_progress_bar.value
        # Increment value by .25
        pbcurrent = self.ids.my_progress_bar.value
        pbcurrent += 25

        current2 += 29
        # If statement to start over after 100

        # Update the progress bar
        self.ids.my_progress_bar.value = pbcurrent

        self.ids.my_progress_bar2.value = current2
        # Update the label
        # self.ids.my_label.text = f'{int(current)}% Progress'

    # see doc MDProgress bar

    """
    What happens when you click on the window (sepcificallly on the image)
    saves the local coordinates of the user's click on the image.
    draws a circle around where the user clicked to inform user of click location 
    Return: None
    """

    def on_touch_up(self, touch):
        # do nothing if video
        if self.currentMediaType == self.video:
            return

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

    def processMedia(self):
        if self.currentMediaType == self.video:
            self.processVideo()
        else:
            self.processImage()

    def processImage(self):
        manual = self.ids.manual_switch.active

        if manual:
            self.manualProcess()
        else:
            self.automaticProcess()

    def processVideo(self):
        clip = VideoFileClip(self.mediaPath)

        duration = clip.duration
        fl = lambda f, t: alignFrame(f(t), t, progress=((t / duration) * 100), interval=5,
                                     progress_bar=self.ids.my_progress_bar,
                                     mirrorX=self.ids.mirrorX_switch.active,
                                     mirrorY=self.ids.mirrorY_switch.active)

        rotatedClip = clip.fl(fl)

        # file_chosen = filechooser.open_file(title="Name Rotated Video")
        # if not file_chosen:
        #     return
        # outfile = file_chosen[0]
        outfile = self.vidPreviewPath
        rotatedClip.write_videofile(outfile, fps=15)
        self.vidPreview()

    def automaticProcess(self):
        src_image = self.currentImg
        scale_factor = min(1280 / src_image.shape[1], 720 / src_image.shape[0])
        img = cv2.resize(src_image, None, fx=scale_factor, fy=scale_factor)
        pbcurrent = self.ids.my_progress_bar.value
        pbcurrent += 12

        self.ids.my_progress_bar.value = pbcurrent

        # horizon_contour = find_horizon(img)
        # # Draw the contour on the image
        # cv2.drawContours(img, [horizon_contour], -1, (0, 255, 0), 2)

        critical_points = find_horizon_point(img, 1, 1, 1, 0.3, 0.7, debug=True)

        # if no critical points found, use default values
        if not critical_points:
            print('no criticalnpoints found')
            cx = img.shape[0] // 2
            cy = img.shape[1] // 2
        else:
            print(critical_points)
            cx = int(critical_points[0])
            cy = int(critical_points[1])

        # check for mirror X and mirror Y settings
        mirrorX = self.ids.mirrorX_switch.active
        mirrorY = self.ids.mirrorY_switch.active

        # scale touch coordinates to image size
        h, w, c = src_image.shape
        ix = cx // scale_factor
        iy = cy // scale_factor
        print(ix, iy)
        # rotate the image and update the preview
        rotatedImage = rotateImage(
            src_image, h, w, c, ix, iy, mirrorX, mirrorY)
        pbcurrent = self.ids.my_progress_bar.value
        pbcurrent = 79

        self.ids.my_progress_bar.value = pbcurrent
        self.updateImage(rotatedImage)
        pbcurrent = 100

        self.ids.my_progress_bar.value = pbcurrent

    """
    Uses the point selected  by the user to equirotate the preview Image
    Return: None
    """

    def manualProcess(self):
        # if no point selected, message the user and return
        if not self.selectedPoint:
            toast("no point selected")
            pbcurrent = self.ids.my_progress_bar.value
            pbcurrent = 0

            self.ids.my_progress_bar.value = pbcurrent
            return

        # remove selected point from image
        self.canvas.remove(self.selectedPoint)
        self.selectedPoint = None

        pbcurrent = 8
        self.ids.my_progress_bar.value = pbcurrent

        src_image = self.currentImg
        imgSize = self.ids.previewImage.size

        # check for mirror X and mirror Y settings
        mirrorX = self.ids.mirrorX_switch.active
        mirrorY = self.ids.mirrorY_switch.active

        # flip Y cordinate
        self.touchLocalY = -(self.touchLocalY - imgSize[1])

        # scale touch coordinates to image size
        h, w, c, ix, iy = scaleImage(
            src_image, imgSize, self.touchLocalX, self.touchLocalY)
        pbcurrent += 12
        self.ids.my_progress_bar.value = pbcurrent
        print(f"Clicked Location (x,y): {ix},{iy}")

        # rotate the image and update the preview
        rotatedImage = rotateImage(
            src_image, h, w, c, ix, iy, mirrorX, mirrorY)
        pbcurrent += 18
        self.ids.my_progress_bar.value = pbcurrent

        self.updateImage(rotatedImage)

    # flips the current image vertically
    def flipVertical(self):
        # do nothing more if video
        if self.currentMediaType == self.video:
            return

        flippedImage = cv2.flip(self.currentImg, 0)
        self.updateImage(flippedImage, flipV_inverse=True)

    # flips the current image horizontally
    def flipHorizontal(self):
        # do nothing more if video
        if self.currentMediaType == self.video:
            return

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
        pbcurrent -= 18
        self.ids.my_progress_bar.value = pbcurrent
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
        pbcurrent = 79
        self.ids.my_progress_bar.value = pbcurrent
        # save the previewImage and update the visual
        cv2.imwrite(self.previewimgPath, newImg, [
            int(cv2.IMWRITE_JPEG_QUALITY), 100])
        self.ids.previewImage.reload()
        pbcurrent = 100
        self.ids.my_progress_bar.value = pbcurrent

    '''
    alignFrame
    takes a bitmap array and video timing data to process into it's equirotated state
    frame: a bitmap array of a frame of a video
    frameNum: the number of the specific frame out of the video
    interval: the interval in which the horizon location will be recalculated after [interval] frames
    '''


def alignFrame(get_frame, t, progress, progress_bar=None, interval=(5 / 15), mirrorX=False, mirrorY=False):
    Clock.async_tick()
    if progress_bar:
        progress_bar.value = int(progress)

    global rotator, shiftx
    frameIntervalProgress = t % interval

    # when at frame interval
    # use the horizon to find the current rotations of the frame
    if frameIntervalProgress == 0:
        rotator, shiftx = getRotator(get_frame, mirrorX, mirrorY)

    # apply rotations to the current frame
    frame = np.roll(get_frame, shiftx, axis=1)
    rotated_image = rotator.rotate(frame)

    if mirrorX and not mirrorY:
        return cv2.flip(rotated_image, 1)
        print("flipped Horizontal")
    if mirrorY and not mirrorX:
        return cv2.flip(rotated_image, 0)
        print("flipped Vertical")
    if mirrorX and mirrorY:
        return cv2.flip(rotated_image, -1)
        print("flipped Both")

    return rotated_image


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

    return rotated_image


"""
takes image frame and returns the equirotate object used to rotate this image
rotator can be used on multiple frames of similar to image used in this function
"""


def getRotator(src_image, mirrorX, mirrorY):
    scale_factor = min(1280 / src_image.shape[1], 720 / src_image.shape[0])
    img = cv2.resize(src_image, None, fx=scale_factor, fy=scale_factor)

    critical_points = find_horizon_point(img, 1, 1, 1, .3, .7)

    # if no critical points found, use default values
    if not critical_points:
        cx = img.shape[0] // 2
        cy = img.shape[1] // 2
    else:
        cx = int(critical_points[0])
        cy = int(critical_points[1])

    # # draw circle on horizon critical point
    # cv2.circle(img, (cx, cy), 10, (200, 0, 0), -1)
    #
    # cv2.imshow("i", img)

    # scale coordinates to image size
    h, w, c = src_image.shape
    ix = cx // scale_factor
    iy = cy // scale_factor
    print(ix, iy)

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

    # rotate (yaw, pitch, roll)
    rotator = EquirectRotate(h, w, (myY, myP, myR))
    return rotator, shiftx


def useRotator(rotator, img):
    rotated_image = rotator.rotate(img)
    return rotated_image


def drawVideoLogo(img):
    midX = img.shape[1] // 2
    midY = img.shape[0] // 2

    height = int(img.shape[0] * .2)
    width = int(img.shape[1] * .2)

    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (midX - (width // 2), midY + (height // 2))

    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (midX + (width // 2), midY - (height // 2))

    # Black color in BGR
    color = (0, 0, 0)

    # Line thickness of -1 will fill shape
    thickness = -1

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(img, start_point, end_point, color, thickness)

    pt1 = start_point[0] + int(width * .2), start_point[1] - int(height * .2)
    pt2 = start_point[0] + int(width * .2), start_point[1] - int(height * .8)
    pt3 = start_point[0] + int(width * .85), start_point[1] - (height // 2)

    triangle_cnt = np.array([pt1, pt2, pt3])
    cv2.drawContours(image, [triangle_cnt], 0, (255, 255, 255), -1)

    return image


class MainScreenApp(MDApp):

    def build(self):
        return MainScreen()


MainScreenApp().run()
