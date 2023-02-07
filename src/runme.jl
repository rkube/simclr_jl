#
using Augmentor
using CairoMakie
#using ImageCore
using Flux
using LinearAlgebra
using MLUtils
using Statistics
using Zygote

push!(LOAD_PATH, "/home/rkube/repos/kstar_ecei_data/")
using kstar_ecei_data

using simclr_jl


"""
    Load ECEI data
"""
shotnr = 26327
data_norm, tbase_norm = get_shot_data(26327; basedir="/home/rkube/datasets/kstar_ecei")
labels = get_labels(26327);
num_samples = size(data_norm)[end]
# Stack data, first, and second derivative
data_trf = zeros(Float32, 24, 8, 3, num_samples-2);
data_trf[:, :, 1, :] = data_norm[:, :, 2:end-1] 
data_trf[:, :, 2, :] = data_norm[:, :, 3:end] .- data_norm[:, :, 1:end-2];
data_trf[:, :, 3, :] = data_norm[:, :, 1:end-2] .- 2f0 * data_norm[:, :, 2:end-1] .+ data_norm[:, :, 3:end];
data_trf = (data_trf .- mean(data_trf, dims=(1, 2, 4))) ./ std(data_trf, dims=(1, 2, 4));


# Define the dataset and set up data loader
n_views = 2
batch_size = 256
τ = 5f-1
num_epochs = 200

pl =  GaussianBlur(3:2:7, 1f0:1f-1:2f0) |> FlipX() * FlipY() * NoOp() |> Rotate(-10:10) |>  CropNative(axes(data_trf)[1:2]);

#
ds_multi = contrastive_ds(data_trf, pl, n_views)

# Get an element from batched observations
loader = DataLoader(ds_multi, batchsize=batch_size, shuffle=false)
bobs = first(loader);
size(bobs) # (24, 8, 3, n_views, batch_size) - (width, height, channels, n_views, batch)

# Squeeze batch observation into 4-d array so that the model can work on them.
bobs_trf = reshape(bobs, size(bobs)[1:3]..., size(bobs, 4) * size(bobs, 5));
# Now the different views of a given sample are consecutive in the last dimension:
for ix_b ∈ 1:batch_size
    for ix_v ∈ 1:n_views
        @assert all(bobs_trf[:, :, :, (ix_b - 1) * n_views + ix_v] .== bobs[:, :, :, ix_v, ix_b])
    end
end


model = Chain(Conv((5, 3), 3 => 32, relu),
              BatchNorm(32),
              MaxPool((2, 1)),
              Conv((5, 3), 32 => 64, relu),
              BatchNorm(64),
              Conv((5, 3), 64 => 64, relu),
              BatchNorm(64),
              Conv((2, 2), 64 => 128, relu),
              BatchNorm(128),
              x -> reshape(x, 128, size(x)[end]), # Now 128 * n_views * batch_size
              # non-linear projection head
              Dense(128 => 128, relu),
              Dense(128 => 32))

params = Flux.params(model)
opt = ADAM(1e-3)

loss_vec = zeros(num_epochs);
# Small-batch overfitting...
bob = first(loader);
bobs_trf = reshape(bobs, size(bobs)[1:3]..., size(bobs, 4) * size(bobs, 5));
for epoch in 1:200
    loss, back = Zygote.pullback(params) do

        # Calculate similarity,
        # sim_i,j = zᵢ' * zⱼ / ||zᵢ|| ||zⱼ||
        zs = model(bobs_trf);

        # The contrastive loss function for a pair is given by
        # ℓ(i,i) = -log[exp(sim(zᵢ,zⱼ) / τ) /  ∑_{k=1}^{2N} [k≠i] exp(sᵢₖ/τ)]

        # To calculate the loss function we need to calculate the matrix S with entries
        # sᵢⱼ = zᵢᵀzⱼ / ||zᵢ|| ||zⱼ||
        z_norm = sqrt.(sum(zs .* zs, dims=1));
        S = exp.(zs' * zs ./ (z_norm' * z_norm) ./ τ);

        # The numerator is for ℓ(i,j) is then simply S[i,j]

        # The loss function for a positive pair (i.e. (2k-1, 2k), (2k, 2k-1) for k=1...N = (1,2), (2,1), (3,4), (4,3), (5,6), (6,5)...
        # is given by ℓ(i,j) = -log [exp(sᵢⱼ/τ) / ∑_{k=1}^{2N} [k≠i] exp(sᵢₖ/τ) ]
        # That is, the denominator for ℓ(2,1) is 
        #  ∑_{k=1}^{2N} [k≠2] exp(s₂ₖ/τ) = ∑_{k=1}^{2N} exp(s₂ₖ/τ) - exp(s₂₂/τ)
        # For ℓ(1,2) the denominator is
        # ∑_{k=1}^{2N} [k≠1] exp(s₁ₖ/τ) = ∑_{k=1}^{2N} exp(s₁ₖ/τ) - exp(s₁₁/τ) 
        # That is, the denominator is the row-wise sum of S subtracted by the diagonal elements.
        denom = sum(S - diagm(diag(S)), dims=2)

        # Generate the index tuples (2k-1, 2k), (2k, 2k-1) used to calculate the loss function
        ix_loss = [(k, iseven(k) ? k-1 : k+1) for k ∈ 1:batch_size]

        L = 0f0
        for i ∈ eachindex(ix_loss)
            L += -log(S[i] / denom[i])
        end

        L 
    end
    grads = back(one(loss))
    

    Flux.update!(opt, params, grads)
    @show epoch, loss
    loss_vec[epoch] = loss;
end

             
# Train a multiclass logistic regression classifier on the representations of the data.
# https://sophiamyang.github.io/DS/optimization/multiclass-logistic/multiclass-logistic.html
num_classes = 2
# Take every second sample. That is because the two different augmented sampels of a single image are
# consecutive in the last dimension
features = model[1:10](bobs_trf)[:, 1:2:end]; 
X̃ = (features .- minimum(features)) ./ (maximum(features) - minimum(features))
hidden_size = 128

# Get labels for first batch. Group class 1 and 2 together
ỹ = labels[1:batch_size]
ỹ[ỹ .== 2] .= 1;
ỹ[ỹ .== 0] .= -1;


"""
    loss(w, X, y)

    Loss function for logistic regression (See Murphy (8.3), p.249)
    w: weights
    X: Feature vector
    y: targets (y∈ ± 1)
"""
loss(W, X, ỹ, λ = 1e-4) = sum(log.(1f0 .+ exp.(-ỹ .* (W' * X)[1,:]))) + λ * W' * W

# Initialize random weights
W = randn(eltype(X̃), hidden_size)


n_steps = 300
W_all = zeros(Float32, length(W), n_steps + 1)
loss_all = zeros(Float32, n_steps + 1)
W_all[:, 1] .= W[:]
τ = 0.9
c = 0.5
λ = 1f-4
α₀ = 1f-1
for s ∈ 1:n_steps
    # Go along the direction p = -grad
    grad, _ = gradient(loss, W_all[:, s], X̃, ỹ, λ)
    # This is ∇fₖᵀ * pₖ in Nocedal&Writght, (3.11) p.41
    m = -grad' * grad
    j = 0
    α = α₀
    while loss(W_all[:, s] .- α * grad, X̃, ỹ, λ) > loss(W_all[:, s], X̃, ỹ, λ) - c * α * m
        α = α * τ
        j = j+1 
        #j > 20 && break
    end
    
    W_all[:, s+1] = W_all[:, s] .- α * grad
    loss_all[s + 1] = loss(W_all[:, s + 1], X̃, ỹ)
    abs(loss_all[s] - loss_all[s+1]) < 1f-3 && break
end

f, a, p = lines(loss_all)

# Make predictions
scatter(1:256, sigmoid(W'*X̃)[1,:])

samples_pos = ỹ .== -1
samples_neg = ỹ .== 1

f = Figure()
a = Axis(f[1, 1])
scatter!(a, (1:256)[samples_pos], sigmoid(W' * X̃[:, samples_pos])[1, :], color=:red, label="positive samples")
scatter!(a, (1:256)[samples_neg], sigmoid(W' * X̃[:, samples_neg])[1, :], color=:blue, label="negative samples")
axislegend()
f





#|Z = -features * W;
P = softmax(Z, dims=2);


size(labels[1:batch_size])



