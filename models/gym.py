from models.brain import Brain
from models.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer()
    trainer.clear_environment()

    trainer.load_data()

    brain = Brain()
    brain.build()

    trainer.train_brain(brain)

    trainer.plot_history()
