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
shotnr = 26512
data_norm, tbase_norm = get_shot_data(26512; basedir="/home/rkube/gpfs/kstar_ecei")
labels = get_labels(26512);
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
τ = 1f-1
num_epochs = 100

#pl =  GaussianBlur(3:2:7, 1f0:1f-1:2f0) |> FlipX() * FlipY() * NoOp() |> Rotate(-10:10) |>  CropNative(axes(data_trf)[1:2]);

#pl =  GaussianBlur(3:2:7, 1f0:1f-1:2f0) |> ColorJitter(0.8:0.1:1.2, -0.2:0.1:0.2) |> Zoom(0.8:0.05:1.2);
pl = GaussianBlur(3:2:7, 1f0:1f-1:2f0) |>  Zoom(0.8:0.05:1.2);
#
ds_multi = contrastive_ds(data_trf, pl, n_views)

# Get an element from batched observations
loader = DataLoader(ds_multi, batchsize=batch_size, shuffle=true, partial=false)
#bobs = first(loader);
#size(bobs) # (24, 8, 3, n_views, batch_size) - (width, height, channels, n_views, batch)

# Squeeze batch observation into 4-d array so that the model can work on them.
#bobs_trf = reshape(bobs, size(bobs)[1:3]..., size(bobs, 4) * size(bobs, 5));
# Now the different views of a given sample are consecutive in the last dimension:
# for ix_b ∈ 1:batch_size
#     for ix_v ∈ 1:n_views
#         @assert all(bobs_trf[:, :, :, (ix_b - 1) * n_views + ix_v] .== bobs[:, :, :, ix_v, ix_b])
#     end
# end


model = Chain(Conv((5, 3), 3 => 64, relu),
              BatchNorm(64),
              MaxPool((2, 1)),
              Conv((5, 3), 64 => 128, relu),
              BatchNorm(128),
              Conv((5, 3), 128 => 256, relu),
              BatchNorm(256),
              Conv((2, 2), 256 => 256, relu),
              BatchNorm(256),
              x -> reshape(x, 256, size(x)[end]), # Now 128 * n_views * batch_size
              # non-linear projection head
              Dense(256 => 1024, relu),
              BatchNorm(1024),
              Dense(1024 => 32, bias=false));
              
              
model = model |> gpu;

params = Flux.params(model)
opt = AdamW(1e-3, (0.9, 0.999), 1e-4)

loss_vec = zeros(num_epochs);
# Small-batch overfitting...

# Generate the index tuples (2k-1, 2k), (2k, 2k-1) used to calculate the loss function
ix_loss = [(k, iseven(k) ? k-1 : k+1) for k ∈ 1:(2*batch_size)]
# The same elements, but as a bool index array
ix_arr = zeros(Bool, 2*batch_size, 2*batch_size)
for ij ∈ ix_loss
    ix_arr[ij...] = true
end

for epoch in 1:num_epochs
    for bobs ∈ loader
        bobs_trf = reshape(bobs, size(bobs)[1:3]..., size(bobs, 4) * size(bobs, 5)) |> gpu;

        loss, back = Zygote.pullback(params) do

            # Calculate similarity,
            # sim_i,j = zᵢ' * zⱼ / ||zᵢ|| ||zⱼ||
            zs = model(bobs_trf);

            # The contrastive loss function for a pair is given by
            # ℓ(i,i) = -log[exp(sim(zᵢ,zⱼ) / τ) /  ∑_{k=1}^{2N} [k≠i] exp(sᵢₖ/τ)]

            # To calculate the loss function we need to calculate the matrix S with entries
            # sᵢⱼ = zᵢᵀzⱼ / ||zᵢ|| ||zⱼ||
            z_norm = sqrt.(sum(zs .* zs, dims=1));
            S = zs' * zs ./ (z_norm' * z_norm) ./ τ;
            # The numerator is for ℓ(i,j) is -log(exp(S)), so simply add -S[i,j] over the indices
            L = -sum(S[ix_arr])

        #    L = 0f0    
        #    for (i, j) ∈ ix_loss
        #        L -= S[i, j];
        #    end
            # The loss function for a positive pair (i.e. (2k-1, 2k), (2k, 2k-1) for k=1...N = (1,2), (2,1), (3,4), (4,3), (5,6), (6,5)...
            # is given by ℓ(i,j) = -log [exp(sᵢⱼ/τ) / ∑_{k=1}^{2N} [k≠i] exp(sᵢₖ/τ) ]
            # That is, the denominator for ℓ(2,1) is 
            #  ∑_{k=1}^{2N} [k≠2] exp(s₂ₖ/τ) = ∑_{k=1}^{2N} exp(s₂ₖ/τ) - exp(s₂₂/τ)
            # For ℓ(1,2) the denominator is
            # ∑_{k=1}^{2N} [k≠1] exp(s₁ₖ/τ) = ∑_{k=1}^{2N} exp(s₁ₖ/τ) - exp(s₁₁/τ) 
            # That is, the denominator is the row-wise sum of S subtracted by the diagonal elements
            # Instead of doing this, we use that exp(-large number) ≈ 0 and add just such a diagonal
            # matrix to S before taking exp of it.
            L = 5f-1 * (L + log(sum(exp.(S .+ (one(S) .* -1f10))))) / batch_size 

            # Log output
            # Find out, how often the resepctive positive example is in the top-1 and top-5 cosine similarity matrix
            # See https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
            Zygote.ignore() do 
                # Fetch the cosine similarity of the matching pair example.
                S_other = S[ix_arr]
                # Right to the row-vector S_other, put all other cosine similarities, but with the cosine similarity
                # of the example to itself and the matching example set to a small negative value
                S2 = copy(S)
                S2[ix_arr] .= -9f5
                S2[diagind(S2)] .= -9f5
                # Stack the matrices next to each other
                S_all = [S_other S2] |> cpu;

                # Find the index with the largest cosine similarity in each row. Index==1 means 
                # that the augmented example has largest similarity.
                ranking_ix = [x[2] for x in argmax(sortslices(S_all, dims=2, rev=true), dims=2)]
                # Top-1 accuracy:
                top1 = mean(ranking_ix .== 1)
                top5 = mean(ranking_ix .<= 5)
                @show top1, top5
            end


            L
        end
        grads = back(one(loss))
        

        Flux.update!(opt, params, grads)
        loss_vec[epoch] += loss;
    end
    loss_vec[epoch] = loss_vec[epoch] / (length(loader) * batch_size)
    @show loss_vec[epoch]
end


# Train a multiclass logistic regression classifier on the representations of the data.
# https://sophiamyang.github.io/DS/optimization/multiclass-logistic/multiclass-logistic.html
num_classes = 2
bs_lr = 16384
# Take every second sample. That is because the two different augmented sampels of a single image are
# consecutive in the last dimension
loader_lr = DataLoader(ds_multi, batchsize=bs_lr, shuffle=false, partial=false)

bobs = first(loader_lr);
# Squeeze batch observation into 4-d array so that the model can work on them.
bobs_trf = reshape(bobs, size(bobs)[1:3]..., size(bobs, 4) * size(bobs, 5)) |> gpu;
features = model[1:10](bobs_trf)[:, 1:2:end] |> cpu;
X̃ = (features .- minimum(features)) ./ (maximum(features) - minimum(features))
hidden_size = 256

# Get labels for first batch. Group class 1 and 2 together
ỹ = labels[1:bs_lr]
ỹ[ỹ .== 2] .= 1;
ỹ[ỹ .== 0] .= -1;


"""
    loss(w, X, y)

    Loss function for logistic regression (See Murphy (8.3), p.249)
    w: weights
    X: Feature vector
    y: targets (y∈ ± 1)
"""
loss_lr(W, X, ỹ, λ = 1e-4) = sum(log.(1f0 .+ exp.(-ỹ .* (W' * X)[1,:]))) + λ * W' * W

# Initialize random weights
W = randn(eltype(X̃), hidden_size)


n_steps = 1000
W_all = zeros(Float32, length(W), n_steps + 1)
loss_all_lr = zeros(Float32, n_steps + 1)
W_all[:, 1] .= W[:]
τ = 0.9
c = 0.5
λ = 1f-4
α₀ = 1f-4
for s ∈ 1:n_steps
    # Go along the direction p = -grad
    grad, _ = gradient(loss_lr, W_all[:, s], X̃, ỹ, λ)
    # This is ∇fₖᵀ * pₖ in Nocedal&Writght, (3.11) p.41
    m = -grad' * grad
    j = 0
    α = α₀
    while loss_lr(W_all[:, s] .- α * grad, X̃, ỹ, λ) > loss_lr(W_all[:, s], X̃, ỹ, λ) - c * α * m
        α = α * τ
        j = j+1 
        #j > 20 && break
    end
    #@show j, loss_lr(W_all[:, s], X̃, ỹ)
    W_all[:, s+1] = W_all[:, s] .- α * grad
    loss_all_lr[s + 1] = loss_lr(W_all[:, s + 1], X̃, ỹ)
    abs(loss_all_lr[s] - loss_all_lr[s+1]) < 1f-3 && break
end

f, a, p = lines(loss_all_lr)

# Make predictions
scatter(1:bs_lr, sigmoid(W'*X̃)[1,:])

samples_pos = ỹ .== -1
samples_neg = ỹ .== 1

f = Figure()
a = Axis(f[1, 1], xlabel="Sample index", ylabel="σ(W * X)")
scatter!(a, (1:bs_lr)[samples_pos], sigmoid(W' * X̃[:, samples_pos])[1, :], color=:red, label="positive samples")
scatter!(a, (1:bs_lr)[samples_neg], sigmoid(W' * X̃[:, samples_neg])[1, :], color=:blue, label="negative samples")
axislegend()

save("plots/logreg_2class.png", f)





#|Z = -features * W;
P = softmax(Z, dims=2);


size(labels[1:batch_size])



