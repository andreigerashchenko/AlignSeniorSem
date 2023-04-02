from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.carousel import Carousel
from kivymd.uix.toolbar import MDTopAppBar
from kivy.uix.image import Image
from kivy.uix.widget import Widget

from kivy.uix.boxlayout import BoxLayout


kv = Builder.load_file('main_screen.kv')


class MainScreen(BoxLayout):

    # replace with the function which does some calculation to maintain progressbar value

    def press_it(self):
        # Grab the current progress bar value
        current = self.ids.my_progress_bar.value
        # Increment value by .25
        current += 25
        # If statement to start over after 100
        if current > 100:
            current = 0
        # Update the progress bar
        self.ids.my_progress_bar.value = current
        # Update the label
        #self.ids.my_label.text = f'{int(current)}% Progress'
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
        #self.ids.my_label2.text = f'{int(current)}% Progress'


class MainScreenApp(MDApp):

    def build(self):
        return MainScreen()


MainScreenApp().run()
