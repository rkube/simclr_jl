#
using Augmentor
using CairoMakie
using MLDatasets
using Statistics
using MLUtils
using ImageCore

push!(LOAD_PATH, "/home/rkube/repos/kstar_ecei_data/")
using kstar_ecei_data

using simclr_jl

# """
#     Define a dataset for ECEI observations where random augmentations are applied to data.
# """
# # plot_img_batch produces 5 contour plots from an RGB image batch
# function plot_img_batch(img_batch)
#     # Convert from RGB batch to array
#     array_batch = permutedims(channelview(img_batch), (2, 3, 1, 4))

#     array_tmp = zeros((24, 5*8))

#     for ix_b ∈ axes(array_batch)[4]
#         array_tmp[:, 1:8] = array_batch[:, end:-1:1, 1, ix_b]
#         array_tmp[:, 17:24] = array_batch[:, end:-1:1, 2, ix_b]
#         array_tmp[:, 33:40] = array_batch[:, end:-1:1, 3, ix_b]
    
#         f = contourf( array_tmp', levels=-0.5:0.1:0.5, colormap=:vik)
#         save("batch_$(ix_b).png", f)
#     end
# end

shotnr = 26327
data_norm, tbase_norm = get_shot_data(26327; basedir="/home/rkube/datasets/kstar_ecei")
num_samples = size(data_norm)[end]
# Alternative idea: Calculate first and second derivative of image time-series manually
# Stack data, first, and second derivative

println("Calculating data features...")
data_trf = zeros(Float32, 24, 8, 3, num_samples-2);
data_trf[:, :, 1, :] = data_norm[:, :, 2:end-1] 
data_trf[:, :, 2, :] = data_norm[:, :, 3:end] .- data_norm[:, :, 1:end-2];
data_trf[:, :, 3, :] = data_norm[:, :, 1:end-2] .- 2f0 * data_norm[:, :, 2:end-1] .+ data_norm[:, :, 3:end];
data_trf = (data_trf .- mean(data_trf, dims=(1, 2, 4))) ./ std(data_trf, dims=(1, 2, 4));


# Define a pipeline
pl = FlipX() * FlipY() * NoOp() |> GaussianBlur(3:2:7, 1f0:1f-1:2f0); # |> Rotate(-10:10) |>  CropNative(axes(data_trf)[1:2]) |>  GaussianBlur(3:2:7, 1f0:1f-1:2f0)
# Instantiate dataset
# my_dset = contrastive_ds(data_trf, pl)

# # Iterate over batches
# loader = DataLoader(my_dset, batchsize=32, shuffle=true)
# for (ix, bobs) ∈ enumerate(loader)
#     @show sum(bobs)
# end


n_views = 5
ds_multi = contrastive_ds(data_trf, pl, n_views)

# Get some observations and check
# 1. That all features (ix_f = 1, 2, 3) have the are transformed identically.
#    This should really be the case, since they are used as different channels in the colorant
# 2. That different observations (vary ix_obs) are all randomly transformed.
res = getobs(ds_multi, 9985:9997)
# STack the together so that all views plot nicely
ix_f = 1 # feature index 1: frame, 2: 1st deriv, 3: 2nd deriv
ix_obs = 1 # observation to look at
all = reduce(vcat, [res[:, end:-1:1, ix_f, v, ix_obs]' for v ∈ axes(res, 4)])

f = Figure()
ax = Axis(f[1, 1])
c = contourf!(ax, all, colormap=:vik)
for s ∈ 1:(n_views - 1)
    lines!(ax, [8 * s, 8 * s], [0, 24], color=:black)
end
f

# Try iterating over ds_multi

loader = DataLoader(ds_multi, batchsize=16)

for (ix, obs) ∈ enumerate(loader)
    @show size(obs)
end
