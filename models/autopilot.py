import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from helpers.image_processor import ImageProcessor
from models.model import Model


class AutoPilot:

    def __init__(self):
        self.socket_io = socketio.Server()
        self.app = Flask(__name__)
        self.model = None

    def load_model(self, path):
        self.model = Model()
        self.model.load(path)

    def telemetry_data(self, sid, data):
        if data is None:
            self.send_control("manual", {})
        else:
            self.process_data(data)

    def connect(self, sid, environ):
        print("Connecting...")
        self.send_control("steer", self.create_steering_data(0, 0))

    def send_control(self, event, data):
        self.socket_io.emit(
            event,
            data=data,
            skip_sid=True
        )

    def create_steering_data(self, steering_angle, throttle):
        return {
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        }

    def drive(self):
        self.socket_io.on('connect', self.connect)
        self.socket_io.on('telemetry', self.telemetry_data)

        self.app = socketio.Middleware(self.socket_io, self.app)
        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)

    def process_data(self, data):
        # steering_angle = float(data["steering_angle"])
        # throttle = float(data["throttle"])
        speed = float(data["speed"])

        image_processor = ImageProcessor()
        image_processor.decode_image(data["image"])
        image_processor.save_image()

        try:
            image = image_processor.process_image()
            steering_angle = self.model.predict_steering_angle(image)
            throttle = 1.0 - steering_angle ** 2 - speed ** 2

            print("steering angle: {}, throttle: {}, speed: {}".format(steering_angle, throttle, speed))
            self.send_control("steer", self.create_steering_data(steering_angle, throttle))

        except Exception as e:
            print(e)
