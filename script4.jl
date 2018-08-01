
include("tools.jl")
using tools

n=200
println("using n=", n)

println()
println("First example with sinus")
dims = [n,n,n]
rhs = rhs_sin(n)
println("Starting approximation")

# calculate the approximation
frames =  Array{Array{Float64,2},1}(3)
rhs_u = unfolding_fun(rhs, 1, dims)
dims_u = [dims[1], prod(dims[2:end]) ]
I,J = aca_fun(rhs_u, dims_u)
C,U,R = cur_fun(rhs_u, dims_u, I, J)
frames[1] = C
dims[1] = length(I)
core = folding(U*R, 1, dims)
for i = 2:3
    core_u = unfolding(core, i)
    I,J = aca(core_u)
    C, U, R = cur(core_u, I, J)
    frames[i] = C
    dims[i] = length(I)
    core = folding(U*R, i, dims)
    gc()
end
ten = tten(core, frames)
rhs_full = get_rhs_sin(200)
err = norm( (rhs_full - fullten(ten))[:])

println("Resulting ranks are: ", size(ten.core))
println("Relative Error: ", err/norm(rhs_full[:]))




println()
println("Second example with norm")
n=200
dims = [n,n,n]
rhs = rhs_norm(n)
println("Starting approximation")

# calculate the approximation
frames =  Array{Array{Float64,2},1}(3)
rhs_u = unfolding_fun(rhs, 1, dims)
dims_u = [dims[1], prod(dims[2:end]) ]
I,J = aca_fun(rhs_u, dims_u)
C,U,R = cur_fun(rhs_u, dims_u, I, J)
frames[1] = C
dims[1] = length(I)
core = folding(U*R, 1, dims)
for i = 2:3
    core_u = unfolding(core, i)
    I,J = aca(core_u)
    C, U, R = cur(core_u, I, J)
    frames[i] = C
    dims[i] = length(I)
    core = folding(U*R, i, dims)
    gc()
end
ten = tten(core, frames)
rhs_full = get_rhs_norm(200)
err = norm( (rhs_full - fullten(ten))[:])

println("Resulting ranks are: ", size(ten.core))
println("Relative Error: ", err/norm(rhs_full[:]))






# ten = approx_aca(rhs, dims)
