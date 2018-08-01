
include("tools.jl")
using tools

n = 200
println("Using n=",n)
xi = get_xi(n)

println()
println("First example with sinus")

rhs = get_rhs_sin(xi)
println("Starting hosvd")
a1, s1 = hosvd(rhs, 1e-4, true)
println("The core tensor has size ",size(a1.core))
residual = rhs - fullten(a1)
err = norm(residual[:])
println("The error ||rhs - full(a1)||_F is: ", err)
println("10⁻⁴ ⋅ ||B₁|| ", 1e-4*norm(rhs[:]))
println("Relative error is: ", err/norm(rhs[:]))

println()
println("Second example with norm")

rhs = get_rhs_norm(xi)
println("Starting hosvd")
a2, s2 = hosvd(rhs, 1e-4, true)
println("The core tensor has size ", size(a2.core))
residual = rhs - fullten(a2)
err = norm(residual[:])
println("The error ||rhs - full(a2)||_F is: ", err)
println("10⁻⁴ ⋅ ||B₂|| ", 1e-4*norm(rhs[:]))
println("Relative Error is: ", err/norm(rhs[:]))

println("creating Plots")
ENV["GKSwstype"]="png"
using Plots;

p1 = scatter(s1[1],  yaxis=(:log))
savefig(p1, "report/eigvalb1.png")

p2 = scatter(s2[1], yaxis=(:log))
savefig(p2, "report/eigvalb2.png")
