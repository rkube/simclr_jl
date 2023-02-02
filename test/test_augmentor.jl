using Augmentor
using CairoMakie

"""
    Explore use of Augmentor
"""

# Get a test image
img = testpattern()
image(img', axis=(yreversed=true, aspect=DataAspect()))

# Define a pipeline that includes some stochasticity
# Random flip and rotation plus cropping to resize to original size
pl = FlipX() * FlipY() * NoOp() |> Rotate([-10.0, -5.0, 0.0, 5.0, 10.0]) |> CropNative(axes(img))

# Plot this twice to verify that we are indeed doing lazy-evaluation
f = Figure()
ax1 = Axis(f[1, 1], aspect=DataAspect(), yreversed=true)
ax2 = Axis(f[1, 2], aspect=DataAspect(), yreversed=true)
image!(ax1, augment(img, pl)')
image!(ax2, augment(img, pl)')
f

# Now let's test the new array type with lazy-evaluation on element access
X_train, _  = MNIST(:train)[:]
batch = X_train[:, :, 1:3]
out = similar(batch)

pl = FlipX() * FlipY() * NoOp() |> Rotate([-30.0, -15.0, 15.0, 30.0]) |> CropNative(axes(X_train[:, :, 1])) |> GaussianBlur(3:5, 1.0:0.1:2.0)

# Apply transformation pipeline to a batch of images.
# Check if the three images have 
augmentbatch!(CPUThreads(), out, batch, pl)
in_out = zeros(28*2, 28*3)
in_out[1:28, 1:end] = reshape(batch, 28, 3*28)
in_out[29:end, 1:end] = reshape(out, 28, 3*28)

image(in_out)