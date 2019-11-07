from models.model import Model
from models.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer()
    trainer.clear_environment()

    trainer.load_data()

    model = Model()
    model.build()

    trainer.train_model(model)

    trainer.plot_history()
