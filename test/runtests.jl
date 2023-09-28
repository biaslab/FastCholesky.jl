using FastCholesky
using Test
using LinearAlgebra
using StaticArrays

make_rand_diagonal(size::Number) = Diagonal(10rand(size))
make_rand_posdef(size::Number) = make_rand_posdef(make_rand_lowertriangular(size))
make_rand_hermitian(size::Number) = Hermitian(make_rand_posdef(size))
make_rand_posdef(L::LowerTriangular) = L * L' + size(L, 1) * I
make_rand_lowertriangular(size) = LowerTriangular(10rand(size, size))

function make_rand_inputs(size)
    inputs = [
        make_rand_diagonal(size), make_rand_posdef(size), make_rand_hermitian(size), 1.0 * I(size)
    ]
    # StaticArrays do not work efficiently for large inputs anyway
    if size < 32
        push!(inputs, SMatrix{size, size}(make_rand_posdef(size)))
    end
    return inputs
end

@testset "FastCholesky.jl" begin

    # General properties
    for size in 1:20:1000
        for input in make_rand_inputs(size)

            # We check only posdef inputs
            @test isposdef(input)

            let input = input
                C = fastcholesky(input)

                @test issuccess(C)
                @test isposdef(C)

                @test collect(fastcholesky(input).L) ≈ collect(cholesky(input).L)
                @test cholinv(input) ≈ inv(input)
                @test cholsqrt(input) ≈ cholesky(input).L
                @test chollogdet(input) ≈ logdet(input)
                @test all(cholinv_logdet(input) .≈ (inv(input), logdet(input)))
                @test cholsqrt(input) * cholsqrt(input)' ≈ sqrt(input) * sqrt(input)'
                @test cholsqrt(input) * cholsqrt(input)' ≈ input
                
            end

            
        end
    end

    @test_throws ErrorException fastcholesky(I)
    @test_throws ErrorException fastcholesky!(I)

    # The functions should work on numbers
    @testset let number = rand()
        @test fastcholesky(number) === number
        @test cholinv(number) ≈ inv(number)
        @test cholsqrt(number) ≈ sqrt(number)
        @test all(cholinv_logdet(number) .≈ (inv(number), logdet(number)))
    end
end
