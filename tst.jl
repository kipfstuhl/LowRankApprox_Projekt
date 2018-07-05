
include("tools.jl")
using tools
using Base.Test

@testset "HOSVD tools" begin
    @testset "Matricisations" begin
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
        
    end;


    @testset "HOSVD" begin
        # set the random number generator to a defined state for
        # reproducibility
        srand(0);
        b = rand(100,100,100);
        ϵ = 1e-4;
        c = hosvd(b, ϵ)

        # for the comparison need to multiply ϵ with norm of b, as the error
        # in this hosvd implementation is relative
        @test norm((fullten(c)-b)[:],2) <= ϵ * norm(b[:],2)
    end;
end;
