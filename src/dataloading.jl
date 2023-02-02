using MLUtils
using Augmentor
using MLUtils
using ImageCore
"""
First attempt at defining a 'dataset' for the KSTAR ECEi data.

Data are accessed from a DataLoader: https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.DataLoader
This requires that a numobs and getobs method is defined for this type.

"""

# # Datatype has the data to operate on and a pipeline
# # 
# struct kstar_ecei_single{T}
#     data_arr::T
#     trf 
# end

# # Define a custom getobs call that transforms one image with the pre-defined pipeline
# function MLUtils.getobs(ds::kstar_ecei_dset, ix)
#     obs = ds.data_arr[:, :, :, ix]; # Fetch a single observation
#     # Transform into colorview. Channels need to be successive. They will be interpreted as the 
#     # channels of colorant RGB 
#     obs_img = colorview(RGB, permutedims(obs, (3, 1, 2)));
#     # Transform image with pipeline
#     obs_trf = augment(obs_img, ds.trf);
#     # Transform RGB image back to array.
#     obs_trf = permutedims(channelview(obs_trf), (2, 3, 1))
# end

# # Define custom getobs for batches. With buffer allocation
# function MLUtils.getobs(ds::kstar_ecei_dset, ix::AbstractVector)
#     batch_dim = [size(ds.data_arr)[[3, 1, 2]]..., length(ix)]
#     buffer = colorview(RGB, zeros(eltype(ds.data_arr), batch_dim...))
#     MLUtils.getobs!(buffer, ds, ix)
# end

# function MLUtils.getobs!(buffer, ds, ix)
#     @show "getobs!", ix
#     batch = ds.data_arr[:, :, :, ix]
#     batch_img = colorview(RGB, permutedims(batch, (3, 1, 2, 4)))
#     augmentbatch!(CPUThreads(), buffer, batch_img, ds.trf)
#     permutedims(channelview(buffer), (2, 3, 1, 4))
# end

# MLUtils.getobs(ds::kstar_ecei_dset) = ds
# MLUtils.numobs(ds::kstar_ecei_dset) = size(ds.data_arr)[end]




"""
    A batch in the SimCLR loader returns a list of num_trf tensors, each of shape [num_batch, num_ch, width, height].
    Modify the getobs method accordingly.

    Introduce a contrastive dataset that returns mulitple, randomly augmented versions of each observation
"""

export contrastive_ds, MLUtils.getobs, MLUtils.numobs

struct contrastive_ds{T}
    data_arr::T
    trf
    n_views::Int
end

MLUtils.getobs(ds::contrastive_ds) = ds
MLUtils.numobs(ds::contrastive_ds) = size(ds.data_arr)[end]

# Define a custom getobs call that transforms one image with the pre-defined pipeline
function MLUtils.getobs(ds::contrastive_ds, ix)
    obs = ds.data_arr[:, :, :, ix]; # Fetch a single observation
    obs_img = colorview(RGB, permutedims(obs, (3, 1, 2))); # Convert single observation to RGB image
    # Allocate an empty array to store ds.n_views augmentations
    obs_trf = Array{RGB{eltype(ds.data_arr)}, 3}(undef, 24, 8, ds.n_views)
  
    # Transform image with pipeline
    for ix_v ∈ axes(obs_trf, 3)
        obs_trf[:, :, ix_v] = augment(obs_img, ds.trf)
    end
    # Transform RGB image back to array.
    obs_trf = permutedims(channelview(obs_trf), (2, 3, 1, 4))
end

# Define custom getobs for batches. With buffer allocation
function MLUtils.getobs(ds::contrastive_ds, ix::AbstractVector)
    # length(ix) is the effective batch_size
    batch_dim = [size(ds.data_arr)[[3, 1, 2]]..., ds.n_views, length(ix)]
    buffer = colorview(RGB, zeros(eltype(ds.data_arr), batch_dim...))
    MLUtils.getobs!(buffer, ds, ix)
end

function MLUtils.getobs!(buffer, ds, ix)
    @show "getobs!", ix
    @show "getobs!", size(buffer)
    batch = ds.data_arr[:, :, :, ix]
    batch_img = colorview(RGB, permutedims(batch, (3, 1, 2, 4)))

    # Augment entire batch to generate a single view
    for ix_v ∈ axes(buffer, 3)
        buffer_view = view(buffer, :, :, ix_v, :)
        augmentbatch!(CPUThreads(), buffer_view, batch_img, ds.trf)
    end
    permutedims(channelview(buffer), (2, 3, 1, 4, 5))
end



