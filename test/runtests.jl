using FastCholesky
using Test
using LinearAlgebra
using StaticArrays
using StaticArraysCore

make_rand_diagonal(size::Number) = Diagonal(10rand(size))
make_rand_posdef(size::Number) = collect(make_rand_hermitian(size))
make_rand_hermitian(size::Number) = make_rand_hermitian(make_rand_lowertriangular(size))
make_rand_hermitian(L::LowerTriangular) = Hermitian(L * L' + (size(L, 1) + 1) * I)
make_rand_lowertriangular(size) = LowerTriangular(10rand(size, size))

function make_rand_inputs(size)
    inputs = [
        make_rand_diagonal(size),
        make_rand_posdef(size),
        make_rand_hermitian(size),
        1.0 * I(size),
    ]
    # StaticArrays do not work efficiently for large inputs anyway
    if size < 32
        push!(inputs, SMatrix{size,size}(make_rand_posdef(size)))
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

                # Check that we do not lose the static type in the process for example
                @test typeof(cholesky(input)) === typeof(fastcholesky(input))
            end
        end

        nonposdefmatrix = Matrix(Diagonal(-ones(size)))

        @test !issuccess(fastcholesky!(nonposdefmatrix))
    end

    @test cholinv(2.0I) ≈ 0.5I
    @test cholsqrt(2.0I) ≈ sqrt(2.0) * I
    @test chollogdet(I) ≈ 0.0
    @test all(cholinv_logdet(1.0I) .≈ (1.0I, 0.0))
    @test_throws ArgumentError chollogdet(2I)
    @test_throws ErrorException fastcholesky(I)
    @test_throws ErrorException fastcholesky!(I)

    # The functions should work on numbers
    let number = rand()
        @test fastcholesky(number) === number
        @test cholinv(number) ≈ inv(number)
        @test cholsqrt(number) ≈ sqrt(number)
        @test chollogdet(number) ≈ logdet(number)
        @test all(cholinv_logdet(number) .≈ (inv(number), logdet(number)))
    end

    @testset "special case #1 (found in ExponentialFamily.jl)" begin
        # This is a very bad matrix, but should be solveable
        F = [
            42491.1429254459 1.0544416413649244e6 64.9016820609457 1712.2779951809016
            1.0544416413649244e6 2.616823794441869e7 1610.468694700484 42488.422800411565
            64.9016820609457 1610.468694700484 0.10421453600353446 2.6155294717625517
            1712.2779951809016 42488.422800411565 2.6155294717625517 69.0045838263577
        ]
        @test inv(fastcholesky(F)) * F ≈ Diagonal(ones(4)) rtol=1e-4
        @test fastcholesky(F).L ≈ cholesky(F).L
    end
end
