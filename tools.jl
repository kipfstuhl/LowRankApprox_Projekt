
module tools

export unfolding, folding, mode_n_mult, tten, hosvd
export get_xi, get_rhs_sin, get_rhs_norm, fullten
export rhs_sin, rhs_norm



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


"""
    unfolding_fun(a, n, dims) -> a_u(i,j)

Returns a function for evaluating the unfolding (matricisation,
flattening) of a tensor that is given only via a function.

"""
function unfolding_fun(a, n, dims)
    # this is just for getting the first index after permutation, it
    # is assumed that the rank of the tensors is not huge
    temp = dims[1:end .!= n]
    ind = temp[1]

    # here one-based indexing is a PITA. Maybe this can be done
    # easier, but at least this works. It has been tested with the
    # array based version.
    if n == 1
        ret = :( (i,j) ->
                 if rem(j,$ind)==0
                 a(i,$ind,div(j,$ind))
                 else
                 a(i,rem(j,$ind),1+div(j,$ind))
                 end
                 )
    elseif n == 2
        ret = :( (i,j) ->
                 if rem(j,$ind)==0
                 a($ind,i,div(j,$ind))
                 else
                 a(rem(j,$ind),i,1+div(j,$ind))
                 end
                 )
    elseif n==3
        ret = :( (i,j) ->
                 if rem(j,$ind)==0
                 a($ind,div(j,$ind),i)
                 else
                 a(rem(j,$ind),1+div(j,$ind),i)
                 end
                 )
    end
    # Show(ret)                   # for seeing what substitutions are made
    return eval(ret)
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

# @inline function ξ(i)
#     return i/(n+1)
# end


function get_rhs_sin(xi::Array{Float64,1})
    n = length(xi)
    rhs = Array{Float64,3}((n,n,n))
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
    rhs = Array{Float64,3}((n,n,n))
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
    rhs = Array{Float64,3}((n,n,n))
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
    rhs = Array{Float64,3}((n,n,n))
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
    rhs = Array{Float64,3}((n,n,n))
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

    frames = Array{Array{Float64,2},1}(d)
    if sv
        svs = Array{Array{Float64,1},1}(d)
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
                rᵢ = j==length(S) ? j : (j+1)
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
        a = folding(diagm(S[1:rᵢ])*V[:,1:rᵢ]',i,dims);
        if sv
            svs[i] = S
        end
        # garbage collection makes life easier and code faster
        gc()
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

    frames = Array{Array{Float64,2},1}(d)
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

    frames = Array{Array{Float64,2},1}(d)
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
        gc()
    end
    return res
end



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
    k = 1
    i = 1
    normk2 = 0.0
    us = []
    vs = []
    # use ↓ this crappy construct to simulate a do-while-loop
    while true
        # maxval is not really used here but comes for free in mymax
        # function
        maxval,j = mymax(abs.(Rk[i,:]),J)

        # reorder execution compared to algorith from lecture
        # now this is not really important anymore, but it does not
        # matter where this is done. For an early (not working)
        # version this was important
        append!(I, i)
        append!(J, j)

        δ = Rk[i,j]
        if abs(δ) < 2eps()
            # not low rank approx anymore ...
            if length(I) == maxrk - 1
                # if this happens, no proper LRA has been calculated
                warn("Return early")
                return sort!(I),sort!(J)
            end
        else
            uk = Rk[:,j]
            vk = Rk[i,:]'/δ
            Rk = Rk - uk*vk
            k = k + 1

            # stopping criterion
            # calculate the parts seperately first interactions of uk,
            # vk with old u and v then the sqared norm, at the end add
            # everything up need the old vectors stored for updates
            push!(us, uk)
            push!(vs, vk)
            temp = 0
            # k-2 due to early incrementation of k
            for l in 1:k-2
                temp += uk'*us[l] * vs[l]*vk'
            end
            #                         ↓ sqared norms
            normk2 = normk2 + 2temp + (uk'*uk)*(vk*vk')

            # check for stopping criterion
            # also if it is not fulfulled check for high rank if rank
            # gets too high just return silently and don't complain
            # after all this is for _low rank_ approximation
            ### maybe it is a good idea to add some special warning or
            ### throw an error if rank exceeds the bounds
            if uk'*uk * vk*vk' <= ϵ*normk2 || k==maxrk
                return sort!(I),sort!(J)
            end
        end                     # if abs(δ)<2eps()
        maxval, i = mymax(abs.(uk), I)
    end
end




"""
    mymax(v, I) -> max,i

Computes the maximal value and the index of v, where i ∉ I. I is some
indexset, every element of I is a valid index for v. (The
implementation assumes that at least the minimal index of I is valid.)
"""
function mymax(v, I)
    tmp = v[1]
    indices = setdiff(1:length(v),I)
    ret = min(indices...)
    for i ∈ indices
        if v[i] > tmp
            tmp = v[i]
            ret = i
        end
    end
    return tmp, ret
end


"""
   cur(a, i, j) -> C,U,R

Compute the CUR approximation given the original matrix a and the
indiex sets i and j.
"""
function cur(a, i, j)
    return a[:,j], inv(a[i,j]), a[i,:]
end



end

