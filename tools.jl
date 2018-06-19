


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
"""
function mode_n_mult(a,n, m)
    dims = size(a)
    new_dims = [dims...]
    new_dims[n] = size(m,1)
    b = m*unfolding(a, n)
    folding(b,n,new_dims)
end
