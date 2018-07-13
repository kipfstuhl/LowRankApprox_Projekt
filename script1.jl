
include("tools.jl")
using tools

n = 200
println("Using n=",n)
xi = get_xi(n)

println()
println("First example with sinus")

rhs = get_rhs_sin(xi)
println("Starting hosvd")
a1 = hosvd(rhs)
println("The core tensor has size ",size(a1.core))


println()
println("Second example with norm")

rhs = get_rhs_norm(xi)
println("Starting hosvd")
a2 = hosvd(rhs)
println("The core tensor has size ", size(a2.core))


# for plotting use new function
rhs = get_rhs_sin(xi)
a1, s1 = hosvd(rhs, 1e-4, true)


rhs = get_rhs_norm(xi)
a2, s2 = hosvd(rhs, 1e-4, true)
