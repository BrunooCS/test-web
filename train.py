from train_vision import main as main_vision
from train_miniVAE import main_miniVAE
from train_rnn import main as main_rnn

if __name__ == '__main__':
    main_vision(0.03, 32, 'car_racing_data_10000.h5', 2, 128, 'trained_models')
    main_miniVAE(0.03, 32, 'car_racing_data_10000.h5', 6, 1.0, 128, 'trained_models')
    main_rnn(0.03, 32, 32, 'car_racing_data_10000.h5', 20, 1, 'trained_models')