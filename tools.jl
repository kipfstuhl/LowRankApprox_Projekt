
module tools

export unfolding, folding, mode_n_mult, tten, hosvd
export get_xi, get_rhs_sin, get_rhs_norm, fullten
export rhs_sin, rhs_norm

export aca, cur, cur_fun
export unfolding_fun, aca_fun, approx_aca


using LinearAlgebra

#### New interface for right-hand side
# This can be used more like functions.
# For any other function on the right hand side,
# just make a non-abstract subtype of rhs and
# implement the desired function as a callable.

# The functions (r::T)(i,j,k) where T <: rhs is some subtype of rhs
# make the objects callable, i.e. one can write
# > r = rhs_sin(100)
# > r(1,35,83)
# this makes the code very convenient.  The Base.getindex
# implementation for rhs is for a more natural indexign via
# > r[1,35,83]
# which returns the same value as the call above.

abstract type rhs end

struct rhs_sin <: rhs
    n::Int
    den::Float64
    rhs_sin(n::Int) = new(n, 1/(n+1))
end

function (r::rhs_sin)(i::Int,j::Int,k::Int)
    # multiplication with reciprocal denominator is much faster than
    # division
    sin((i+j+k)*r.den)
end


struct rhs_norm <: rhs
    n::Int
    den::Float64
    rhs_norm(n::Int) = new(n, 1/(n+1))
end

function (r::rhs_norm)(i::Int,j::Int,k::Int)
    sqrt( (i*r.den)^2 + (j*r.den)^2 + (k*r.den)^2)
end


function Base.getindex(r::rhs, i::Int, j::Int, k::Int)
    # cheap boundscheck; note this does not use the Julia convetnions
    # of bounds checking with several nested function calls
    # check may be removed for better performance
    correct = 1<=i<=r.n && 1<=j<=r.n && 1<=k<=r.n
    if ~correct
        throw(BoundsError(r,(i,j,k)))
    end
    return r(i,j,k)
end


# alternative to getindex for callables This makes arrayindexing with
# Fortran syntax possible, i.e. a(i,j,k)
#### use with care!!
#### uncomment only if you know what you do
# function (a::Array)(i::Int...)
#     a[i...]
# end


"""
    unfolding_fun(a, n, dims) -> a_u(i,j)

Returns a function for evaluating the unfolding (matricisation,
flattening) of a tensor that is given only via a function.

"""
function unfolding_fun_expr(a, n, dims)
    # this is just for getting the first index after permutation, it
    # is assumed that the rank of the tensors is not huge
    # This is a bit easy here, because I assume tensors of fixed order
    # 3 only. Then products just vanish.
    temp = dims[1:end .!= n]
    ind = Tuple(temp)

    # here one-based indexing is a PITA. Maybe this can be done
    # easier, but at least this works. It has been tested with the
    # array based version.
    if n == 1
        ret = :( (i,j) ->
                 # $a(i, ind2sub($ind,j)...)
                 $a(i, CartesianIndices($ind)[j][1], CartesianIndices($ind)[j][2])
                 # if rem(j,$ind)==0
                 # a(i,$ind,div(j,$ind))
                 # else
                 # a(i,rem(j,$ind),1+div(j,$ind))
                 # end
                 )
    elseif n == 2
        ret = :( (i,j) ->
                 $a(CartesianIndices($ind)[j][1], i, CartesianIndices($ind)[j][2])
                 # if rem(j,$ind)==0
                 # a($ind,i,div(j,$ind))
                 # else
                 # a(rem(j,$ind),i,1+div(j,$ind))
                 # end
                 )
    elseif n==3
        ret = :( (i,j) ->
                 $a(CartesianIndices($ind)[j][1], CartesianIndices($ind)[j][2], i)
                 # if rem(j,$ind)==0
                 # a($ind,div(j,$ind),i)
                 # else
                 # a(rem(j,$ind),1+div(j,$ind),i)
                 # end
                 )
    end
    # show(ret)                   # for seeing what substitutions are made
    return ret
end

function unfolding_fun(a, n, dims)
    return eval(unfolding_fun_expr(a, n, dims))
end


"""
    get_xi(n::Int)
returns a vector xi of length n with entries
```julia
xi[i] = 1/(n+1)
```
"""
function get_xi(n::Int)
    collect(1:n)/(n+1)
end


@inline function ξ(i, n)
    i/(n+1)
end


function get_rhs_sin(xi::Array{Float64,1})
    n = length(xi)
    rhs = Array{Float64,3}(undef, (n,n,n))
    @inbounds for i in 1:n
        for j in 1:n
            @simd for k in 1:n
                rhs[k,j,i] = sin(xi[k]+xi[j]+xi[i])
            end
        end
    end
    return rhs
end

function get_rhs_sin(n::Int)
    rhs = Array{Float64,3}(undef, (n,n,n))
    @inbounds for i in 1:n
        for j in 1:n
            @simd for k in 1:n
                rhs[k,j,i] = sin(ξ(k,n)+ξ(j,n)+ξ(i,n))
            end
        end
    end
    return rhs
end


function get_rhs_sin_opt(n::Int)
    # this function retruns different values
    # this is due to machine arithmetic and
    # the precomputation of the denominator
    rhs = Array{Float64,3}(undef, (n,n,n))
    den::Float64 = 1.0/(n+1)
    @inbounds for i in 1:n
        for j in 1:n
            @simd for k in 1:n
                rhs[k,j,i] = sin((k+j+i)*den)
            end
        end
    end
    return rhs
end


function get_rhs_norm(n::Int)
    rhs = Array{Float64,3}(undef, (n,n,n))
    @inbounds for i in 1:n
        for j in 1:n
            @simd for k in 1:n
                rhs[k, j, i] = sqrt(ξ(k,n)*ξ(k,n)+ξ(j,n)*ξ(j,n)+ξ(i,n)*ξ(i,n))
            end
        end
    end
    return rhs
end

function get_rhs_norm(xi::Array{Float64,1})
    n = length(xi)
    rhs = Array{Float64,3}(undef, (n,n,n))
    @inbounds for i in 1:n
        for j in 1:n
            @simd for k in 1:n
                rhs[k,j,i] = sqrt(xi[k]*xi[k] + xi[j]*xi[j] + xi[i]*xi[i])
            end
        end
    end
    return rhs
end

function approx_sin(n)
    rhs = get_rhs_sin(n)
    hosvd(rhs, 1e-4)
end

function approx_norm(n)
    rhs = get_rhs_norm(n)
    hosvd(rhs, 1e-4)
end



"""
n-mode unfolding or matricisation of a tensor a
"""
function unfolding(a, n)
    dims = size(a)
    indices = setdiff(1:ndims(a),n)
    reshape(permutedims(a,[n;indices]),dims[n],prod(dims[indices]))
end

"""
fold matrix into tensor by mode n
reverse of unfolding
"""
function folding(m, n, dims)
    indices = setdiff(1:length(dims), n)
    a = reshape(m, (dims[n],dims[indices]...))
    permutedims(a,invperm([n;indices]))
end



"""
mode-n multiplication of tensor a with matrix m

# inputs
- a :  Array
- n :  mode for multiplication
- m :  matrix that gets multiplied with a in mode n
"""
function mode_n_mult(a, n, m)
    dims = size(a)
    # new_dims = [dims...]
    new_dims = collect(dims)
    new_dims[n] = size(m,1)
    b = m*unfolding(a, n)
    folding(b,n,new_dims)
end

"""
Type for Tucker tensor
core:   Array, this is the core tensor of size (r₁, r₂, r₃, ...)
frames: Array of Arrays, these are the mode frames
"""
struct tten{T<:AbstractFloat}
    core::Array{T}
    frames::Array{Array{T,2},1}
    
end


"""
    hosvd(a, ϵ)

Computes the Higher Order Singular Value Decomposition

The computation is done via sequentially truncated HOSVD, this is much faster,
as the tensor gets compressed early.
Returns a `tten` object. The tten is a tucker tensor consisting of a
core tensor and an array of mode frames.

input:
- a: full tensor
- ϵ: relative error bound

output:
- `tten` object representing the Tucker approximation
"""
function hosvd(a, ϵ::T=1e-4, sv::Bool=false) where {T<:AbstractFloat}
    d = ndims(a)
    # dims = [size(a)...]
    dims = collect(size(a))
    ϵ = ϵ/sqrt(d) * norm(a[:],2) # divide relative error equal to all dimensions
    ϵ² = ϵ*ϵ                     # use the sqare for cheaper comparison

    frames = Array{Array{Float64,2},1}(undef, d)
    if sv
        svs = Array{Array{Float64,1},1}(undef, d)
    end
    for i in 1:d
        U,S,V = svd(unfolding(a,i))
        rᵢ = 1
        temp = 0.0
        for j in length(S):-1:1
            temp += S[j]*S[j]
            if temp > ϵ²
                # take the one satisfying the error bound, but stay
                # within the bounds => check whether on bound or not
                rᵢ = j==length(S) ? j : (j)
                break
            end
        end
        frames[i] = U[:, 1:rᵢ]
        #a = mode_n_mult(a,i,frames[i]')
        # use a more compact representation of the remaining matrix
        # the implementation of mode_n_mult copys the data around more than once
        # this works as the remainder S*V is just the same as
        # U'*A = U'*U*A*V', as U is orthogonal
        dims[i] = rᵢ
        a = folding(diagm(0=>S[1:rᵢ])*V[:,1:rᵢ]',i,dims);
        if sv
            svs[i] = S
        end
        # garbage collection makes life easier and code faster
        GC.gc()
    end
    if sv
        return tten(a,frames), svs
    else
        return tten(a,frames)
    end
end


# the hosvd2 functions are worse because they reduce the core at the
# end when done during the first loop, simultaneous to svd's, it
# reduces the size and thus cost much
function hosvd2(a, ranks::Vector{T}) where {T<:Int}
    d = ndims(a)
    @assert(d==length(ranks), "dimension mismatch")

    frames = Array{Array{Float64,2},1}(undef, d)
    for i in 1:d
        U = svd(unfolding(a,i))
        frames[i] = U[:, 1:ranks[i]]
    end

    for i in 1:d
        a = mode_n_mult(a,i,frames[i]')
    end
    return tten(a, frames)
    
end

function hosvd2(a, ranks::Tuple{Vararg{T}}) where {T<:Int}
    # rank_arr = [ranks...]
    rank_arr = collect(ranks)
    hosvd(a, rank_arr)
end

function hosvd2(a, tol::T) where {T<:AbstractFloat}
    d = ndims(a)
    tol = tol/sqrt(d) * norm(a[:],2)

    frames = Array{Array{Float64,2},1}(undef, d)
    for i in 1:d
        U,S = svd(unfolding(a,i))
        r_i = length(S)
        for j in 1:length(S)
            # rework this
            # this is not exactly what we want
            # better is ||S[j:end]|| < tol or even
            # sum(S[j:end].^2) < tol^2
            if S[j] < tol
                r_i = j
                break
            end
        end
        frames[i] = U[:, 1:r_i]
        gc()
    end

    for i in 1:d
        a = mode_n_mult(a,i,frames[i]')
        gc()
    end
    return tten(a,frames)
end



function fullten(a::tten{T}) where {T<:AbstractFloat}
    res = a.core
    for i in 1:length(a.frames)
        res = mode_n_mult(res, i, a.frames[i])
        GC.gc()
    end
    return res
end



# """
#     approx_aca(a, dims, ϵ)

# Compute an approximation of a in Tucker format. dim is an array
# holding the size of a.
# """
# function approx_aca(a, dims, ϵ::Float64=1e-4)
#     d = length(dims)
#     frames =  Array{Array{Float64,2},1}(d)
    
#     a_u = unfolding_fun(a, 1, dims)
#     dims_u = [dims[1], prod(dims[2:end]) ]
#     I,J = aca_fun(a_u, dims_u, ϵ)
#     C,U,R = cur_fun(a_u, dims_u, I, J)
#     frames[1] = C
#     dims[1] = length(I)
#     core = folding(U*R, 1, dims)
    
#     for i = 2:3
#         core_u = unfolding(core, i)
#         I,J = aca(core_u, ϵ)
#         C, U, R = cur(core_u, I, J)
#         frames[i] = C
#         dims[i] = length(I)
#         core = folding(U*R, i, dims)
#         gc()
#     end

#     return tten(core, frames)
# end


"""
    aca(a, ϵ)

Computes adaptive cross approximation of matrix a with relative accuracy ϵ.

This implementation uses partial pivoting, this implies the array has
to be evaluated along a row or column. For the computation not the
whole array is needed.

"""
function aca(a::Array{Float64,2},ϵ::Float64=1e-4)
    Rk = a
    maxrk = min(size(a)...)
    I = Int[]
    J = Int[]
    I_good = Int[]
    J_good = Int[]
    k = 1
    i = 1
    # i = 70
    normk2 = 0.0
    us = []
    vs = []
    ϵ = ϵ*ϵ
    # use ↓ this crappy construct to simulate a do-while-loop
    while true
        # maxval is not really used here but comes for free in mymax
        # function
        # maxval,j = mymax(abs.(Rk[i,:]),J)
        # use better pivoting compared to pseudo-alg from lecture
        i,j = find_pivot(Rk, i, I, J)
        # reorder execution compared to algorith from lecture
        # now this is not really important anymore, but it does not
        # matter where this is done. For an early (not working)
        # version this was important
        append!(I, i)
        append!(J, j)

        δ = Rk[i,j]
        # improve the error here. ϵ is maybe not optimal, as it is
        # just the sqared error bound for the overall
        # approximation. This just works for the tested cases.
        if abs(δ) < ϵ
            if length(I) == maxrk - 1
                # if this happens, no proper LRA has been calculated
                @warn("Return with maximal rank => no Low Rank Approximation")
                return I_good,J_good
            end

            # try again, if still bad exit note: this is a trade off
            # between accuracy and speed.
            # It may happen, that there are still good candidates, but
            # those cannot be reached with the search used for the
            # pivot elements. E.g. for block diagonal matrices this
            # easily happens.
            i,j = find_pivot(Rk, i, I, J)
            δ = Rk[i,j]
            if abs(δ) < ϵ
                @warn("Could not find good pivot anymore.")
                return I_good,J_good
            end
        else
            uk = Rk[:,j]
            vk = Rk[i,:]'/δ
            Rk = Rk - uk*vk
            k = k + 1
            # use only the indices that contribute
            append!(I_good, i)
            append!(J_good, j)
            push!(us, uk)
            push!(vs, vk)

            # stopping criterion
            # calculate the parts seperately
            # first interactions of uk, vk with old u and v then the
            # sqared norm, at the end add everything up
            temp = 0
            # k-2 due to early incrementation of k
            for l in 1:k-2
                temp += uk'*us[l] * vs[l]*vk'
            end
            #                         ↓ sqared norms
            normk2 = normk2 + 2temp + (uk'*uk)*(vk*vk')

            # check for stopping criterion
            # also if it is not fulfilled check for high rank, if rank
            # gets too high just return silently and don't complain
            ### maybe it is a good idea to add some special warning or
            ### throw an error if rank exceeds the bounds, after all
            ### this is for _low rank_ approximation
            if uk'*uk * vk*vk' <= ϵ*normk2 || k==maxrk
                return I_good,J_good
            end
        end
        # if abs(δ)<2eps() does not make a
        # difference, if uk and vk are not updated, the error estimate
        # remains the same. So this can be checked in the else
        # statement
        
        # needed when directly implementing pseudo-algo from lecture
        # that is not the best choice
        # maxval, i = mymax(abs.(uk), I)
    end
end


function find_pivot(a, i, I, J, n::Int=3)
    j = 1
    for l = 1:n
        maxval, j = mymax(abs.(a[i,:]), J)
        maxval, i = mymax(abs.(a[:,j]), I)
    end
    return i,j
end


"""
    mymax(v, I) -> max,i

Computes the maximal value and the index of v, where i ∉ I. I is some
indexset, every element of I is a valid index for v. (The
implementation assumes that at least the minimal index of I is valid.)
"""
function mymax(v, I)
    indices = setdiff(1:length(v),I)
    ind = min(indices...)
    maxval = v[ind]
    for i ∈ indices
        if v[i] > maxval
            maxval = v[i]
            ind = i
        end
    end
    return maxval, ind
end






"""
    update_rk(a, I, J) -> r

takes a function a(i,j), arrays if indices I, J and returns function
calculating the residual a - u*s*v'. Where u are columns J of a, v are
rows I of a, and s is inverse of a(I,J).
"""
function update_rk(a, I, J)
    # check this again! Not sure this works
    #
    # Idea: compute s = inv(a(I,J))
    # then compute a - u*s*v'
    # for the last computation use only the rows and columns that are
    # required
    # this function returns a closure, if not used to it looks strange
    # at first

    if length(I) != length(J)
        error("I and J have to be of equal size")
        return
    end
    
    s = zeros(length(I),length(J))
    for (j,jj) ∈ enumerate(J)
        for (i,ii) ∈ enumerate(I)
            s[i,j] = a(ii,jj)
        end
    end
    s = inv(s)

    function u(ii)
        # check sizes!
        ret = zeros(length(J))
        for (j,jj) ∈ enumerate(J)
            ret[j] = a(ii,jj)
        end
        return ret
    end

    function v(jj)
        ret = zeros(length(I))
        for (i,ii) ∈ enumerate(I)
            ret[i] = a(ii,jj)
        end
        return ret
    end

    # define the closure
    function rk(ii,jj)
        return a(ii,jj) - u(ii)'*s*v(jj)
    end

    return rk
end

"""
   find_pivot_fun(a, dims, i, I, J)

retunr indices i,j that maximise a(i,j) using a heuristic (search
along axes of a). dims array contains the size of a, i.e.
a ∈ ℝ^(dims[1],dims[2]).

"""
function find_pivot_fun(a, dims, i, I, J)
    j = 1
    for n = 1:3
        a1 = jj -> abs(a(i,jj))
        # maxval, j = mymax_fun( jj-> abs(a(i,jj)), dims[2], J)
        maxval, j = mymax_fun(a1, dims[2], J)
        a2 = ii -> abs(a(ii,j))
        # maxval, i = mymax_fun( ii-> abs(a(ii,j)), dims[1], I)
        maxval, i = mymax_fun(a2, dims[1], I)
    end
    return i,j
end


"""
    mymax_fun(v, n, I)

return maximum value of function v and index ∈ {1...n}∖I
"""
function mymax_fun(v, n, I)
    indices = setdiff(1:n, I)
    ind = min(indices...)
    # maxval = v(ind)
    maxval = Base.invokelatest(v, ind)
    for i ∈ indices
        # if v(i) > maxval
        if Base.invokelatest(v, i) > maxval
            # maxval = v(i)
            maxval = Base.invokelatest(v, i)
            ind = i
        end
    end
    return maxval, ind
end






function aca_fun(a, dims, ϵ::Float64=1e-4)
    #R = Array{Any,1}([a])
    # R = []
    # push!(R, a)
    # Rk = a
    maxrk = min(dims...)
    I = Int[]
    J = Int[]
    I_good = Int[]
    J_good = Int[]
    k = 1
    i = 1
    normk2 = 0.0
    us = []
    vs = []
    Rk = []
    push!(Rk, a)
    ϵ = ϵ*ϵ
    
    while true
        i,j = find_pivot_fun(Rk[k], dims, i, I, J)
        append!(I, i)
        append!(J, j)

        # δ = R[k](i,j)
        δ = Rk[k](i,j)
        if abs(δ) < ϵ
            # if abs(δ) == 0
            #     return sort!(I),sort!(J)
            # end
            if length(I) == maxrk-1
                @warn("Return with maximal rank")
                return I_good, J_good
            end

            i,j = find_pivot_fun(Rk[k], dims, i, I, J)
            δ = Rk[k](i,j)
            if abs(δ) < ϵ
                @warn("Could not find good pivot aymore.")
                return I_good, J_good
            end
        else
            # uk = (ii)->R[k](ii,j)
            # vk = (jj)->R[k](i,jj)/δ
            uk = (ii)->Rk[k](ii,j)
            vk = (jj)->Rk[k](i,jj)/δ
            push!(us, uk)
            push!(vs, vk)

            # FIX: calculate the whole function in every step
            # this costs a loop of length k in each step
            # push!(R, (ii,jj)->R[k](ii,jj)-uk(ii)*vk(jj) )
            # Rk = update_rk(Rk, uk, vk)
            # Rk = update_rk(a, us, vs)

            # update later, otherwise uk and vk get zero!!!!!
            # Rk = update_rk(a, I, J)
            # k = k+1

            temp = 0.0
            for l in 1:k-2
                # temp += prod(map( ii->uk(ii)*us[l](ii), 1:dims[1])) *
                #     prod(map( jj->ns[l](jj)*vk(jj), 1:dims[2]))
                # use the handy |> piping syntax
                temp += sum(1:dims[1] .|> ii->uk(ii)*us[l](ii)) *
                    sum(1:dims[2] .|> jj->vs[l](jj)*vk(jj))
            end

            # calculate product of sqared norms
            # sqnormprod = prod(map( ii->uk(ii)*uk(ii), 1:dims[1])) * prod(map( jj->vk(jj)*vk(jj), 1:dims[2]))
            sqnormprod = sum(1:dims[1] .|> ii->uk(ii)*uk(ii)) * sum(1:dims[2] .|> jj->vk(jj)*vk(jj))
            normk2 = normk2 + 2temp + sqnormprod
            
            if sqnormprod <= ϵ*normk2 || k==maxrk
                return I_good, J_good
            end

            # use only the indices that contribute to the solution
            append!(I_good, i)
            append!(J_good, j)

            # update here, otherwise uk and vk would be zero
            # k already increased
            k = k+1
            push!(Rk, update_rk(a, I_good, J_good))
        end

    end
end

"""
    cur_fun(a, dims, I, J) -> c, u, r

returns CUR decomposition of `a` given as function a(i,j). dims array
defines sizes of a, i.e. a ∈ ℝ^(dims[1],dims[2]). I, J are the row and
columns indices as returned from aca_fun.

When using aca with functions for huge sizes be careful. This function creates **full matrices**!

"""
function cur_fun(a, dims, I, J)
    m = dims[1]
    n = dims[2]
    c = zeros(m, length(J))
    u = zeros(length(I),length(J))
    r = zeros(length(I), n)

    # set c to coloumns of a
    for (j,jj) ∈ enumerate(J)
        for i in 1:m
            c[i,j] = a(i,jj)
        end
    end

    # set r to rows of a
    for j in 1:n
        for (i,ii) ∈ enumerate(I)
            r[i,j] = a(ii,j)
        end
    end

    # first get a[I,J], then invert it
    for (j,jj) ∈ enumerate(J)
        for (i,ii) ∈ enumerate(I)
            u[i,j] = a(ii,jj)
        end
    end
    u = inv(u)

    return c, u, r
end

"""
   cur(a, i, j) -> C,U,R

Compute the CUR approximation given the original matrix a and the
indiex sets i and j.
"""
function cur(a, i, j)
    return a[:,j], inv(a[i,j]), a[i,:]
end






"""
    approx_aca(a, dims, ϵ)

Compute an approximation of a in Tucker format. dim is an array
holding the size of a.
"""
function approx_aca(a, dims, ϵ::Float64=1e-4)
    d = length(dims)
    frames =  Array{Array{Float64,2},1}(undef, d)
    
    a_u = unfolding_fun(a, 1, dims)
    dims_u = [dims[1], prod(dims[2:end]) ]
    I,J = aca_fun(a_u, dims_u, ϵ)
    C,U,R = cur_fun(a_u, dims_u, I, J)
    frames[1] = C
    dims[1] = length(I)
    core = folding(U*R, 1, dims)
    
    for i = 2:3
        core_u = unfolding(core, i)
        I,J = aca(core_u, ϵ)
        C, U, R = cur(core_u, I, J)
        frames[i] = C
        dims[i] = length(I)
        core = folding(U*R, i, dims)
        gc()
    end

    return tten(core, frames)
end






end

