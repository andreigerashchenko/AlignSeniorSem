from main_screen import MainScreen
from main_screen import MainScreenApp
from kivy import Config
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

# get all the screens using builder and import
from start_screen import StartScreen
Builder.load_file('start_screen.kv')
Builder.load_file('main_screen.kv')


# Define our different screens
class startScreen(Screen):
    root = StartScreen()


class mainScreen(Screen):
    pass



class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)
        self.add_widget(mainScreen(name='mainScreen'))
        # Add the start screen to the screen manager
        self.add_widget(startScreen(name='startScreen'))

        self.current = 'startScreen'
        print('current after init:', self.current)



# Designate Our .kv design file
kv = Builder.load_file('manager.kv')


class fullApp(App):
    def build(self):
        return kv


if __name__ == '__main__':
    fullApp().run()
