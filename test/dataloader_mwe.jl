using MLUtils
using Random
using Augmentor
using ImageCore


struct my_dset{T}
    data_arr::T
    trf
end


num_samples = 100
num_channels = 3
width = height = 32
d = randn(Float32, width, height, num_channels, num_samples)


pl = FlipX() * FlipY() * NoOp() |> GaussianBlur(3:2:5, 1f0:1f-1:2f0)
ds = my_dset(d, pl)


function MLUtils.getobs(dset::my_dset, ix::Int)
    obs = dset.data_arr[:, :, :, ix]
    obs_c = colorview(RGB, permutedims(obs, (3, 1, 2)))
    obs_trf = augment(obs_c, dset.trf)
    permutedims(channelview(obs_trf), (2, 3, 1))
end

MLUtils.numobs(data::my_dset) = size(data.data_arr)[end]

loader = DataLoader(ds, batchsize=-1)

for (ix, obs) ∈ enumerate(loader)
    @show ix, sum(obs)
end



"""
    Now do batches. The docs https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.BatchView say:
    For BatchView to work on some data structure, the type of the given variable data must implement 
    the data container interface. See ObsView for more info.

    So let's take a look at ObsView: https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.ObsView


    For ObsView to work on some data structure, the desired type MyType must implement the following interface:

    getobs(data::MyType, idx) : Should return the observation(s) indexed by idx. In what form is up to the user. Note that idx can be of type Int or AbstractVector.

    numobs(data::MyType) : Should return the total number of observations in data

    So let's do that.
"""

# We defined getobs for integer indices above. Now do AbstractVector
# This is for batches
function MLUtils.getobs(dset::my_dset, ix::AbstractVector)
    # Get the size of the dataset array, sans the number of batches. THat is defined by 
    # the length of the index vector
    @show "getobs", ix
    batch_dim = [size(ds.data_arr)[[3, 1, 2]]..., length(ix)]
    buffer = colorview(RGB, zeros(eltype(dset.data_arr), batch_dim...))
    MLUtils.getobs!(buffer, dset, ix)
end

# getobs(data::MyType) : By default this function is the identity function. If that is not the behaviour that you want for your type, you need to provide this method as well.
MLUtils.getobs(dset::my_dset) = dset

# obsview(data::MyType, idx) : If your custom type has its own kind of subset type, you can return it here. An example for such a case are SubArray for representing a subset of some AbstractArray.

# getobs!(buffer, data::MyType, [idx]) : Inplace version of getobs(data, idx). If this method is provided for MyType, then eachobs can preallocate a buffer that is then reused every iteration. Note: buffer should be equivalent to the return value of getobs(::MyType, ...), since this is how buffer is preallocated by default.
function MLUtils.getobs!(buffer, dset, ix)
    batch = dset.data_arr[:, :, :, ix]
    batch_img = colorview(RGB, permutedims(batch, (3, 1, 2, 4)))
    #out = colorview(RGB, zeros(eltype(batch), size(batch)[[3, 1, 2, 4]]))
    #out = similar(batch_img)
    augmentbatch!(CPUThreads(), buffer, batch_img, dset.trf)
    permutedims(channelview(buffer), (2, 3, 1, 4))
end


# v = ObsView(ds, 1:10);
# v = BatchView(ds, batchsize=19)

# for (ix, bobs) ∈ enumerate(v)
#     @show ix, size(bobs)
# end


loader_batch = DataLoader(ds, batchsize=27, shuffle=true)
for (ix, bobs) ∈ enumerate(loader_batch)
    @show ix, size(bobs)
end
