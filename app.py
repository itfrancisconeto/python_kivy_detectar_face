import threading
from functools import partial
import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

class MainScreen(Screen):
    pass

class Manager(ScreenManager):
    pass

Builder.load_string('''
<MainScreen>:
    FloatLayout:
        Label:
            text: "DETECTOR DE FACES"
            pos_hint: {"x":0.0, "y":0.85}
            size_hint: 1.0, 0.2
        Image:
            id: vid
            size_hint: 1, 0.8
            allow_stretch: True  # allow the video image to be scaled
            keep_ratio: True  # keep the aspect ratio so people don't look squashed
            pos_hint: {'center_x':0.5, 'top':0.9}
''')

class Main(App):
    def build(self):
        self.title = 'Interface com Kivy'
        threading.Thread(target=self.facedetect, daemon=True).start()
        sm = ScreenManager()
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)
        return sm

    def facedetect(self):
        self.do_vid = True
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cam = cv2.VideoCapture(0)
        while (self.do_vid):
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            for (x, y, w, h) in faces:                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            Clock.schedule_once(partial(self.display_frame, frame))

    def display_frame(self, frame, dt):        
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.main_screen.ids.vid.texture = texture

if __name__ == '__main__':
    Main().run()