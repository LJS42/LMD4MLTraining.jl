# Data Loader
function get_data_loader()
    preprocess(x, y) = (reshape(x, 28, 28, 1, :), Flux.onehotbatch(y, 0:9))
    x_train_raw, y_train_raw = MLDatasets.MNIST.traindata()
    x_train, y_train = preprocess(x_train_raw, y_train_raw)
    return Flux.DataLoader((x_train, y_train); batchsize=128, shuffle=true)
end
# Model
function get_model()
    return Chain(
        Conv((5, 5), 1 => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(256, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
    )
end

function main()
    # Define quantities to track
    quantities = [LossQuantity(), GradNormQuantity(), DistanceQuantity(), UpdateSizeQuantity(), NormTestQuantity()]

    model = get_model()
    data_loader = get_data_loader()
    optim = Flux.setup(Adam(3f-4), model)
    loss_fn(ŷ, y) = vec(Flux.logitcrossentropy(ŷ, y; agg=identity))
    learner = Learner(model, data_loader, loss_fn, optim, quantities)

    train_leaner!(learner, 1, false)

    println("Training finished.")
end

main()
