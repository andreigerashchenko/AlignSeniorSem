from kivy import Config
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

# Define our different screens


class startScreen(Screen):
	pass


class mainScreen(Screen):
	pass


class pref(ScreenManager):
	pass


class WindowManager(ScreenManager):
	pass


# Designate Our .kv design file
kv = Builder.load_file('manager.kv')


class fullApp(App):
	def build(self):
		return kv


if __name__ == '__main__':
	fullApp().run()
