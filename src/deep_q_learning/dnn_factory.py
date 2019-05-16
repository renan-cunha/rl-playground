import keras
from keras.layers import Dense


def create_dnn(input_length: int, num_hidden_neurons: int,
               num_hidden_layers: int, output_length: int,
               learning_rate: float) -> keras.models:
    model = keras.models.Sequential()
    model.add(Dense(num_hidden_neurons, input_dim=input_length,
                    activation='relu'))

    for i in range(num_hidden_layers):
        model.add(Dense(num_hidden_neurons, activation="relu"))

    model.add(Dense(output_length, activation="linear"))
    model.compile(loss="mse",
                  optimizer=keras.optimizers.Adam(lr=learning_rate))
    return model


