
include("tools.jl")
using tools

# let the JIT Compiler do some work
n=50
dims = [n,n,n]
rhs = rhs_norm(n)
rhs = rhs_sin(n)
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




for n = [200,300,400]
    println("=====================")
    println("using n=", n)

    println()
    println("First example with sinus")
    dims = [n,n,n]
    rhs = rhs_sin(n)
    println("Starting approximation")

    tic()
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
    toc()
    ten = tten(core, frames)
    rhs_full = get_rhs_sin(n)
    err = norm( (rhs_full - fullten(ten))[:])

    println("Resulting ranks are: ", size(ten.core))
    println("Relative Error: ", err/norm(rhs_full[:]))


    println()
    println("Second example with norm")
    dims = [n,n,n]
    rhs = rhs_norm(n)
    println("Starting approximation")
    tic()
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
    toc()
    ten = tten(core, frames)
    rhs_full = get_rhs_norm(n)
    err = norm( (rhs_full - fullten(ten))[:])

    println("Resulting ranks are: ", size(ten.core))
    println("Relative Error: ", err/norm(rhs_full[:]))
end


for n = [500,600,700]
    println("=====================")
    println("using n=", n)

    println()
    println("First example with sinus")
    dims = [n,n,n]
    rhs = rhs_sin(n)
    println("Starting approximation")

    tic()
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
    toc()
    ten = tten(core, frames)
    println("Resulting ranks are: ", size(ten.core))


    println()
    println("Second example with norm")
    dims = [n,n,n]
    rhs = rhs_norm(n)
    println("Starting approximation")
    tic()
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
    toc()
    ten = tten(core, frames)
    println("Resulting ranks are: ", size(ten.core))
end



gc()

