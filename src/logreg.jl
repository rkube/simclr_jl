#
using LinearAlgebra
using CairoMakie
using Zygote


# Test logistic regression
# We want to find the minimum of the function f2.
# Hint: it's at (1, 1)
# Adding a term for L2 weight regularization (8.30)
f2(θ, λ) = 0.5 * (θ[1] * θ[1] - θ[2])^2 + 0.5 * (θ[1] - 1.0)^2 + λ * θ' * θ

function ∇f2(θ, λ) 
    ∇ = zeros(2)
    ∇[1] = 0.5 * 2 * (θ[1] * θ[1] - θ[2]) * 2 * θ[1] + (θ[1] - 1)
    ∇[2] = -(θ[1] * θ[1] - θ[2])
    ∇ .+ 2 * λ * θ
end

xrg = 0.0:0.1:2.0
yrg = -0.5:0.1:3.0
coord_grid = Iterators.product(xrg, yrg)

# Test gradient descent method in Murphy, 8.3.2, p.250
# Learning rate
η = 0.5
# https://en.wikipedia.org/wiki/Backtracking_line_search
# Search control parameters
τ = 0.95
α = η

n_steps = 30
θ_vals = zeros(Float64, 2, n_steps+1)

for s ∈ 1:n_steps
    α = 0.5 # re-set α
    # Run backtracking line search to get optimal step rate
    #
    p = -∇f2(θ_vals[:, s])
    m = ∇f2(θ_vals[:, s])' * p
    j = 0
    t = -c*m
    while (f2(θ_vals[:, s]) - f2(θ_vals[:, s] .+ α*p)) ≥ α*t
        j = j+1
        α = τ * α
        j > 10 && break
    end
    @show α, j, t
    θ_vals[:, s+1] = θ_vals[:, s] .- α * ∇f2(θ_vals[:, s])
    #@show θnext, θ
    #θ = θnext
end




############### Now do the same with zygote
# See if the gradient calculated by zygote matches the manual definition
gradient(f2, [0.5, 0.1], 1e-4)[1] ≈ ∇f2([0.5, 0.1], 1e-4)
λ = 1e-4

for s ∈ 1:n_steps
    grad, _ = gradient(f2, θ_vals[:, s], λ)
    m = grad' * grad
    j = 0
    t = -c*m
    α = 0.5
    while(f2(θ_vals[:, s]) - f2(θ_vals[:, s] .+ α * p)) ≥ α*t
        j = j+1
        α = τ * α
        j > 10 && break
    end
    @show α, j
    θ_vals[:, s+1] = θ_vals[:, s] .- α * grad
end

f = Figure()
a = Axis(f[1, 1])
contour!(a, xrg, yrg, map(x -> f2([x[1], x[2]], 1e-4), coord_grid), levels=32)
scatter!(a, θ_vals[1, :], θ_vals[2, :], marker=:circle, color=:red)
lines!(a, θ_vals[1, :], θ_vals[2, :], color=:red)
f



