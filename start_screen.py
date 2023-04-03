from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

kv = Builder.load_file('start_screen.kv')


class StartScreen(BoxLayout):
    def oFile(self):
        self.ids.openFile.source = 'images/whitefileclicked.png'
        self.ids.hh.text = 'button'

    def sFile(self):
        self.ids.openFile.source = 'images/whitefile.png'
        self.ids.hh.text = 'Hey Horizon'

    def oFolder(self):
        self.ids.openFolder.source = 'images/folderDown.png'
        self.ids.hh.text = 'opening Folder'

    def sFolder(self):
        self.ids.openFolder.source = 'images/folder.png'
        self.ids.hh.text = 'Hey Horizon'

    def oProject(self):
        self.ids.openProject.source = 'images/projectDown.png'
        self.ids.hh.text = 'opening Project'

    def sProject(self):
        self.ids.openProject.source = 'images/project.png'
        self.ids.hh.text = 'Hey Horizon'


class StartScreenApp(App):
    def build(self):
        #Window.borderless = True
        return StartScreen()


if __name__ == '__main__':
    StartScreenApp().run()
