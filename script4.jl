
include("tools.jl")
using Main.tools
using LinearAlgebra


# let the JIT Compiler do some work
# global n=50
# global dims = [n,n,n]
# global rhs = rhs_norm(n)
# global rhs = rhs_sin(n)
# global frames =  Array{Array{Float64,2},1}(undef, 3)
# global rhs_u = unfolding_fun(rhs, 1, dims)
# global dims_u = [dims[1], prod(dims[2:end]) ]
# global I,J = aca_fun(rhs_u, dims_u)
# global C,U,R = cur_fun(rhs_u, dims_u, I, J)
# frames[1] = C
# global dims[1] = length(I)
# global core = folding(U*R, 1, dims)
# for i = 2:3
#     # global core_u = unfolding(core, i)
#     # global I,J = aca(core_u)
#     # global C, U, R = cur(core_u, I, J)
#     # frames[i] = C
#     # dims[i] = length(I)
#     # global core = folding(U*R, i, dims)
#     # GC.gc()
#     I, J, C, U, R = do_it(core, dims, i)
# end

function do_calc(core_fun, dims)
    frames =  Array{Array{Float64,2},1}(undef, 3)
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
        C,U,R = cur(core_u, I, J)
        frames[i] = C
        dims[i] = length(I)
        core = folding(U*R, i, dims)
        GC.gc()
    end
    return (core, frames, dims)
end
    
n=50
dims = [n,n,n]
rhs = rhs_norm(n)
rhs = rhs_sin(n)

core, frames, dims = do_calc(rhs, dims)
println(dims)


# n=50
# dims = [n,n,n]
# rhs = rhs_norm(n)
# rhs = rhs_sin(n)
# frames =  Array{Array{Float64,2},1}(undef, 3)
# rhs_u = unfolding_fun(rhs, 1, dims)
# dims_u = [dims[1], prod(dims[2:end]) ]
# I,J = aca_fun(rhs_u, dims_u)
# C,U,R = cur_fun(rhs_u, dims_u, I, J)
# frames[1] = C
# dims[1] = length(I)
# core = folding(U*R, 1, dims)
# for i = 2:3
#     # global core_u = unfolding(core, i)
#     # global I,J = aca(core_u)
#     # global C, U, R = cur(core_u, I, J)
#     # frames[i] = C
#     # dims[i] = length(I)
#     # global core = folding(U*R, i, dims)
#     # GC.gc()
#     I, J, C, U, R = do_it(core, dims, i)
# end




for n = [200,300,400]
    println("=====================")
    println("using n=", n)

    println()
    println("First example with sinus")
    global dims = [n,n,n]
    global rhs = rhs_sin(n)
    println("Starting approximation")

    # tic()
    start = time_ns()
    # calculate the approximation
    # global frames =  Array{Array{Float64,2},1}(undef, 3)
    # global rhs_u = unfolding_fun(rhs, 1, dims)
    # global dims_u = [dims[1], prod(dims[2:end]) ]
    # global I,J = aca_fun(rhs_u, dims_u)
    # global C,U,R = cur_fun(rhs_u, dims_u, I, J)
    # frames[1] = C
    # dims[1] = length(I)
    # global core = folding(U*R, 1, dims)
    # for i = 2:3
    #     global core_u = unfolding(core, i)
    #     global I,J = aca(core_u)
    #     global C, U, R = cur(core_u, I, J)
    #     frames[i] = C
    #     dims[i] = length(I)
    #     global core = folding(U*R, i, dims)
    #     GC.gc()
    # end
    core, frames, dims = do_calc(rhs, dims)
    # toc()
    elapsed_time = time_ns() - start
    println("elapsed time: ", elapsed_time*1e-9)
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
    # tic()
    start = time_ns()
    # calculate the approximation
    # frames =  Array{Array{Float64,2},1}(undef, 3)
    # rhs_u = unfolding_fun(rhs, 1, dims)
    # dims_u = [dims[1], prod(dims[2:end]) ]
    # global I,J = aca_fun(rhs_u, dims_u)
    # global C,U,R = cur_fun(rhs_u, dims_u, I, J)
    # frames[1] = C
    # dims[1] = length(I)
    # global core = folding(U*R, 1, dims)
    # for i = 2:3
    #     global core_u = unfolding(core, i)
    #     global I,J = aca(core_u)
    #     global C, U, R = cur(core_u, I, J)
    #     frames[i] = C
    #     dims[i] = length(I)
    #     global core = folding(U*R, i, dims)
    #     GC.gc()
    # end
    core, frames, dims = do_calc(rhs, dims)
    # toc()
    elapsed_time = time_ns() - start
    println("elapsed time: ", elapsed_time*1e-9)
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

    # tic()
    start = time_ns()
    # calculate the approximation
    # frames =  Array{Array{Float64,2},1}(undef, 3)
    # rhs_u = unfolding_fun(rhs, 1, dims)
    # dims_u = [dims[1], prod(dims[2:end]) ]
    # global I,J = aca_fun(rhs_u, dims_u)
    # global C,U,R = cur_fun(rhs_u, dims_u, I, J)
    # frames[1] = C
    # dims[1] = length(I)
    # global core = folding(U*R, 1, dims)
    # for i = 2:3
    #     core_u = unfolding(core, i)
    #     global I,J = aca(core_u)
    #     global C, U, R = cur(core_u, I, J)
    #     frames[i] = C
    #     dims[i] = length(I)
    #     global core = folding(U*R, i, dims)
    #     GC.gc()
    # end
    core, frames, dims = do_calc(rhs, dims)
    # toc()
    elapsed_time = time_ns() - start
    println("elapsed time: ", elapsed_time*1e-9)
    ten = tten(core, frames)
    println("Resulting ranks are: ", size(ten.core))


    println()
    println("Second example with norm")
    dims = [n,n,n]
    rhs = rhs_norm(n)
    println("Starting approximation")
    # tic()
    start = time_ns()
    # calculate the approximation
    # frames =  Array{Array{Float64,2},1}(undef, 3)
    # rhs_u = unfolding_fun(rhs, 1, dims)
    # dims_u = [dims[1], prod(dims[2:end]) ]
    # global I,J = aca_fun(rhs_u, dims_u)
    # global C,U,R = cur_fun(rhs_u, dims_u, I, J)
    # frames[1] = C
    # dims[1] = length(I)
    # global core = folding(U*R, 1, dims)
    # for i = 2:3
    #     core_u = unfolding(core, i)
    #     global I,J = aca(core_u)
    #     global C, U, R = cur(core_u, I, J)
    #     frames[i] = C
    #     dims[i] = length(I)
    #     global core = folding(U*R, i, dims)
    #     GC.gc()
    # end
    core, frames, dims = do_calc(rhs, dims)
    # toc()
    elapsed_time = time_ns() - start
    println("elapsed time: ", elapsed_time*1e-9)
    ten = tten(core, frames)
    println("Resulting ranks are: ", size(ten.core))
end



GC.gc()

