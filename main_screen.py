from kivy.graphics import Color, Ellipse
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
from kivy.uix.image import Image
from kivymd.uix.toolbar import MDTopAppBar
from kivy.uix.carousel import Carousel
from kivymd.app import MDApp
from time import sleep
from kivy.config import Config
from kivy.lang import Builder

from plyer import filechooser

Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '2048')
# Config.set('graphics', 'height', '1280')
kv = Builder.load_file('main_screen.kv')


class HelpPopup(Popup):
    pass


class PrefPopup(Popup):
    pass


class MainScreen(BoxLayout):
    popup = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__()
        self.touchLocalX = None
        self.touchLocalY = None
        self.selectedPoint = None

    def openFileBrowser(self):
        # save original directory to restore at end
        cwd = os.getcwd()

        file_path = ""
        file_path = filechooser.open_file(title="File Selection", filters=[
            "*.jpg", "*.png", "*.jpeg"])
        if file_path == []:
            pass
        else:
            self.ids.previewImage.source = file_path[0]

        # restore original directory
        os.chdir(cwd)

    def open_Help(self):
        self.popup = HelpPopup()
        self.popup.open()

    def open_popup(self):
        self.popup = PrefPopup()
        self.popup.open()

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

    def on_touch_up(self, touch):

        # touch provides the click's global cords for the entire app
        # localize the touch coordinates to the image
        previewImg = self.ids.previewImage
        imgCords = previewImg.pos
        imgSize = previewImg.size

        # get coordinates of mouse click relative to bottom left of image
        touchLocalX = touch.x - imgCords[0]
        touchLocalY = touch.y - imgCords[1]

        # if click is outside of image bounds, disregard it
        if touchLocalX < 0 or touchLocalX > imgSize[0]:
            return
        if touchLocalY < 0 or touchLocalY > imgSize[1]:
            return

        self.touchLocalX = touchLocalX
        self.touchLocalY = touchLocalY

        # draw cricle to represent selected area
        with self.canvas:
            # remove selected point from image
            if self.selectedPoint:
                self.canvas.remove(self.selectedPoint)
                self.selectedPoint = None

            Color(.75, .3, .3, .6)
            d = 15.
            self.selectedPoint = Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))

    def processImage(self):
        # placeholder for how manual vs automatic processing
        # takes place
        manual = True
        if manual:
            self.manualProcess()

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

        # flip Y cordinate if mirrorY is active
        if self.ids.mirrorY_switch.active:
            self.touchLocalY = -(self.touchLocalY - imgSize[1])

        # get image paths for input and output
        src_path = previewImg.source
        opfile = ".previewImg.jpg"

        # open the image to be transformed
        src_image = cv2.imread(src_path)
        print(src_path)

        # scale touch coordinates to image size
        h, w, c, ix, iy = scaleImage(
            src_image, imgSize, self.touchLocalX, self.touchLocalY)
        print(f"Clicked Location (x,y): {ix},{iy}")

        # rotate the image and update the preview
        rotatedImage = rotateImage(src_image, h, w, c, ix, iy, mirrorX, mirrorY)

        print(opfile)
        print(cv2.imwrite(opfile, rotatedImage, [int(cv2.IMWRITE_JPEG_QUALITY), 100]))
        previewImg.source = opfile
        src_path = opfile

        previewImg.reload()


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

    # mirror axis WIP
    if mirrorX and not mirrorY:
        flipAxis = 1
    if mirrorY and not mirrorX:
        flipAxis = 0
    if mirrorX and mirrorY:
        flipAxis = -1

    if not (mirrorX or mirrorY):
        final_image = rotated_image
    else:
        final_image = cv2.flip(rotated_image, flipAxis)

    return final_image


class MainScreenApp(MDApp):

    def build(self):
        return MainScreen()


MainScreenApp().run()
