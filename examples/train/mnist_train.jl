using Statistics
using Flux
using MLDatasets 
using GLMakie

function preprocess(x, y)
    # Add singleton color-channel dimension to features for Conv-layers
    x = reshape(x, 28, 28, 1, :)   # (H, W, C, N)

    # One-hot encode targets
    y = Flux.onehotbatch(y, 0:9)

    return x, y
end

function accuracy(model)
    # Use onecold to return class index
    ŷ = Flux.onecold(model(x_test))
    y = Flux.onecold(y_test)

    return mean(ŷ .== y)
end

#-------------------------------------------------------------------------
# Data preprocessing and plotting
# ------------------------------------------------------------------------

x_train_data, y_train_data = MLDatasets.MNIST.traindata()
x_test_data,  y_test_data  = MLDatasets.MNIST.testdata()

x_train, y_train = preprocess(x_train_data, y_train_data)
x_test,  y_test  = preprocess(x_test_data,  y_test_data)

#plot one training sample
i = 10
img = x_train[:, :, 1, i]        # 28×28 matrix
label = y_train_raw[i]

f = Figure()
ax = Axis(f[1, 1], title = "Digit: $label", aspect = DataAspect())
image!(ax, reverse(img,dims =2)) #reverse image so it is shown upright
f

#-------------------------------------------------------------------------
# Data loading and model setup
# ------------------------------------------------------------------------
batchsize = 128 

train_loader = Flux.DataLoader((x_train, y_train);
               batchsize = batchsize, shuffle=true)
model = Chain(
            Conv((5, 5), 1 => 6, relu),  # 1 input color channel
            MaxPool((2, 2)),
            Conv((5, 5), 6 => 16, relu),
            MaxPool((2, 2)),
            Flux.flatten,
            Dense(256, 120, relu),
            Dense(120, 84, relu),
            Dense(84, 10),  # 10 output classes
        )
loss_fn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
optim = Flux.setup(Adam(3.0f-4), model)

#-------------------------------------------------------------------------
# Model training 
# ------------------------------------------------------------------------
run_training = true

if run_training
    losses = Float32[]

    # Train for 5 epochs
    for epoch in 1:5

        # Iterate over batches returned by data loader
        for (i, (x, y)) in enumerate(train_loader)

            # Compute loss and gradients of model w.r.t. its parameters
            loss, grads = Flux.withgradient(m -> loss_fn(m(x), y), model)

            # Update optimizer state
            Flux.update!(optim, model, grads[1])

            # Keep track of losses by logging them in `losses`
            push!(losses, loss)

            # Every fifty steps, evaluate the accuracy on the test set
            # and print the accuracy and loss
            if isone(i) || iszero(i % 50)
                acc = accuracy(model) * 100
                @info "Epoch $epoch, step $i:\t loss = $(loss), acc = $(acc)%"
            end
        end
    end
end