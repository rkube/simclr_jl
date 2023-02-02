
using Augmentor
using MLUtils
using Random
using simclr_jl


"""
    Tests for
        * Instantiation of contrastive dataset
        * Correct dispatching to extended definitions in src/dataloading.jl
        * Working definition of getobs for integer index
        * Working definition of getobs for index vector
        * Correct sizes of batches when iterating over DataLoader
"""

n_obs = 7
n_views = 5
pipeline = FlipX() * FlipY() * NoOp()
ds = contrastive_ds(randn(16, 16, 3, n_obs), pipeline, n_views)

@show numobs(ds)
@show getobs(ds)


res = getobs(ds, 1)
@show size(res)

res = getobs(ds, 2:4)
@show size(res)

loader = DataLoader(ds, batchsize=2, shuffle=true)

for (ix, bobs) ∈ enumerate(loader)
    @show size(bobs)
end

