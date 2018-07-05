

include("tools.jl")
using tools
using Base.Test

a = collect(1:8);
a = reshape(a, (2,2,2));

@test unfolding(a, 1) == [1 3 5 7;
                          2 4 6 8]

@test unfolding(a, 2) == [1 2 5 6;
                          3 4 7 8]

@test unfolding(a, 3) == [1 2 3 4;
                          5 6 7 8]

@test a == folding(unfolding(a,1),1,size(a))
@test a == folding(unfolding(a,2),2,size(a))
@test a == folding(unfolding(a,3),3,size(a))


b = rand(100,100,100);
c = hosvd(b)

@test norm((fullten(c)-b)[:],2) <= 1e-4 * norm(b[:],2)
