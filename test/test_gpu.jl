# test_gpu.jl
println("using CUDA")
@time using CUDA
println("using Flux")
@time using Flux

println("model = Chain(Conv((5, 3), 3 => 32, relu),...")

@time model = Chain(Conv((5, 3), 3 => 32, relu),
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
              Dense(128 => 32));

println("model_g = model |> gpu")
@time model_g = model |> gpu


S = randn(Float32, 16, 16) |> gpu

diagm(diag((S))


