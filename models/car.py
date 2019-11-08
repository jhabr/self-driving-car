import os
from models.autopilot import AutoPilot

if __name__ == '__main__':
    auto_pilot = AutoPilot()
    auto_pilot.load_brain(os.path.join(os.getcwd(), '..', 'artefacts', 'model.h5'))
    auto_pilot.drive()
