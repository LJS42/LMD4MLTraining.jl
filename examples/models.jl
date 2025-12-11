"""Base model: CNN arquitecture to perform image classification on MNIST dataset
"""
using Flux 

cnn_model = Chain(
            Conv((5, 5), 1 => 6, relu),  # 1 input color channel
            MaxPool((2, 2)),
            Conv((5, 5), 6 => 16, relu),
            MaxPool((2, 2)),
            Flux.flatten,
            Dense(256, 120, relu),
            Dense(120, 84, relu),
            Dense(84, 10),  # 10 output classes
        )
