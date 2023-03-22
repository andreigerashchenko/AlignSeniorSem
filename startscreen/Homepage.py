from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle


class MyFloatLayout(FloatLayout):
    def on_size(self, instance, size):
        return True

    def __init__(self, **kwargs):
        super(MyFloatLayout, self).__init__(**kwargs)
        with self.canvas.before:
            # Set the background color to blue
            Color(0.2, 0.6, 0.9, 1.0)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        if (self.on_size):
            self.bind(size=self._update_rect, pos=self._update_rect)

    def on_image_touch(self, instance, touch):
        if instance.collide_point(*touch.pos):
            print('Image touched at', touch.pos)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    pass


class HomepageApp(App):

    def build(self):
        Clock.max_iteration = 1
        #Clock.schedule_interval(game.update, 1.0 / 60.0)
        return MyFloatLayout()


if __name__ == '__main__':
    HomepageApp().run()
