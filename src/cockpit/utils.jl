function track!(sess::Session, loss::LossQuantity, model, x, y)
    ŷ = model(x)
    l = compute_loss(loss, ŷ, y)
    sess.output = [sess.output; l]
    return l
end
