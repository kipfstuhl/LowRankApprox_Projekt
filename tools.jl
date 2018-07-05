
module tools

export unfolding, folding, mode_n_mult, tten, hosvd
export get_xi, get_rhs_sin, get_rhs_norm, fullten

"""
some definitions
"""
function get_xi(n)
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

function get_rhs_sin(n::Integer)
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

"""
this function retruns different values
this is due to machine arithmetic and the optimisation
"""
function get_rhs_sin_opt(n::Integer)
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


function get_rhs_norm(n::Integer)
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
    #reshape(PermutedDimsArray(a,[n;indices]),dims[n],prod(dims[indices]))
end

"""
fold matrix into tensor by mode n
reverse of unfolding
"""
function folding(m, n, dims)
    indices = setdiff(1:length(dims), n)
    a = reshape(m, (dims[n],dims[indices]...))
    permutedims(a,invperm([n;indices]))
    #PermutedDimsArray(a,invperm([n;indices]))
end



"""
mode-n multiplication of tensor a with matrix m
"""
function mode_n_mult(a,n, m)
    dims = size(a)
    # new_dims = [dims...]
    new_dims = collect(dims)
    new_dims[n] = size(m,1)
    b = m*unfolding(a, n)
    folding(b,n,new_dims)
end


struct tten{T<:AbstractFloat}
    core::Array{T}
    frames::Array{Array{T,2},1}

    
    
end

a²

"""
Higher Order Singular Value Decomposition
"""
function hosvd(a, ϵ::T=1e-4) where {T<:AbstractFloat}
    d = ndims(a)
    # dims = [size(a)...]
    dims = collect(size(a))
    ϵ = ϵ/sqrt(d) * norm(a[:],2) # divide relative error equal to all dimensions
    ϵ² = ϵ*ϵ                     # use the sqare for cheaper comparison

    frames = Array{Array{Float64,2},1}(d)
    for i in 1:d
        U,S,V = svd(unfolding(a,i))
        rᵢ = 1
        temp = 0.0
        for j in length(S):-1:1
            temp += S[j]*S[j]
            if temp > ϵ²
                # take the one satisfying the error bound, but stay
                # within the bounds => check whether on bound or not
                rᵢ = j==length(S)?j:(j+1)
                break
            end
        end
        frames[i] = U[:, 1:rᵢ]
        #a = mode_n_mult(a,i,frames[i]')
        # use a more compact representation of the matrix
        # the implementation of mode_n_mult copys the data around
        dims[i] = rᵢ
        a = folding(diagm(S[1:rᵢ])*V[:,1:rᵢ]',i,dims); 
        gc()                    # garbage collection makes life easier
    end

    return tten(a,frames)
end


# the hosvd2 functions are worse because they reduce the core at the
# end when done during the first loop, simultaneous to svd's, it
# reduces the size and thus cost much
function hosvd2(a, ranks::Vector{T}) where {T<:Integer}
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

function hosvd2(a, ranks::Tuple{Vararg{T}}) where {T<:Integer}
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

end
