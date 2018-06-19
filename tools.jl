
module tools

export unfolding, folding, mode_n_mult, tten, hosvd
export get_xi, get_rhs_sin, get_rhs_norm

"""
some definitions
"""
function get_xi(n)
    collect(1:n)/(n+1)
end

function get_rhs_sin(xi)
    n = length(xi)
    rhs = Array{Float64,3}((n,n,n))
    for i in 1:n
        for j in 1:n
            for k in 1:n
                rhs[i,j,k] = sin(xi[i]+xi[j]+xi[k])
            end
        end
    end
    return rhs
end

function get_rhs_norm(xi)
    n = length(xi)
    rhs = Array{Float64,3}((n,n,n))
    for i in 1:n
        for j in 1:n
            for k in 1:n
                rhs[i,j,k] = sqrt(xi[i]*xi[i] + xi[j]*xi[j] + xi[k]*xi[k])
            end
        end
    end
    return rhs
end

function approx_sin(n)
    xi = get_xi(n)
    rhs = get_rhs_sin(xi)
    hosvd(rhs, 1e-4)
end

function approx_norm(n)
    xi = get_xi(n)
    rhs = get_rhs_norm(xi)
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
    new_dims = [dims...]
    new_dims[n] = size(m,1)
    b = m*unfolding(a, n)
    folding(b,n,new_dims)
end


type tten{T<:Real}
    core::Array{T}
    frames::Array{Array{T,2},1}

    
    
end


"""
Higher Order Singular Value Decomposition
"""
function hosvd(a, ranks::Vector{T}) where {T<:Integer}
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

function hosvd(a, ranks::Tuple{Vararg{T}}) where {T<:Integer}
    rank_arr = [ranks...]
    hosvd(a, rank_arr)
end

function hosvd(a, tol::T) where {T<:AbstractFloat}
    d = ndims(a)
    tol = tol/sqrt(d) * norm(a[:],2)

    frames = Array{Array{Float64,2},1}(d)
    for i in 1:d
        U,S = svd(unfolding(a,i))
        r_i = length(S)
        for j in 1:length(S)
            if S[j] < tol
                r_i = j
                break
            end
        end
        frames[i] = U[:, 1:r_i]
    end

    for i in 1:d
        a = mode_n_mult(a,i,frames[i]')
    end
    return tten(a,frames)
end

end
