using BenchmarkTools, LinearAlgebra
using FastCholesky

const SUITE = BenchmarkGroup()
const BenchmarkFloatTypes = (Float32, Float64, BigFloat)

for T in BenchmarkFloatTypes
    SUITE[string(T)] = BenchmarkGroup()
    SUITE[string(T)]["Number"] = BenchmarkGroup()
    SUITE[string(T)]["Matrix"] = BenchmarkGroup()
    SUITE[string(T)]["Diagonal"] = BenchmarkGroup()
end

# general benchmarks
for T in BenchmarkFloatTypes

    # number
    a = rand(T)

    SUITE[string(T)]["Number"]["fastcholesky"] = @benchmarkable fastcholesky($a)
    SUITE[string(T)]["Number"]["fastcholesky!"] = @benchmarkable fastcholesky!($a)
    SUITE[string(T)]["Number"]["cholinv"] = @benchmarkable cholinv($a)
    SUITE[string(T)]["Number"]["cholsqrt"] = @benchmarkable cholsqrt($a)
    SUITE[string(T)]["Number"]["chollogdet"] = @benchmarkable chollogdet($a)
    SUITE[string(T)]["Number"]["cholinv_logdet"] = @benchmarkable cholinv_logdet($a)

    for dim in (2, 5, 10, 20, 50, 100, 200, 500)

        SUITE[string(T)]["Matrix"]["dim:" * string(dim)] = BenchmarkGroup()

        # Skip `BigFloat` benchmarks for large dimensions because they are extremely slow
        if (T == BigFloat) && (dim >= 100)
            continue
        end

        # generate positive-definite matrix of specified dimensions
        A = randn(T, dim, dim)
        B = A * A' + dim * I(dim)
        C = similar(B)

        # The inplace version works only for `Matrix{<:BlasFloat}`
        if T <: LinearAlgebra.BlasFloat
            SUITE[string(T)]["Matrix"]["dim:" * string(dim)]["fastcholesky!"] = @benchmarkable fastcholesky!($C)
        end

        # `sqrt` does not work for `BigFloat` inputs
        if T != BigFloat
            SUITE[string(T)]["Matrix"]["dim:" * string(dim)]["cholsqrt"] = @benchmarkable cholsqrt($B)
        end

        # define benchmarks
        SUITE[string(T)]["Matrix"]["dim:" * string(dim)]["fastcholesky"] = @benchmarkable fastcholesky($B)
        SUITE[string(T)]["Matrix"]["dim:" * string(dim)]["cholinv"] = @benchmarkable cholinv($B)
        SUITE[string(T)]["Matrix"]["dim:" * string(dim)]["chollogdet"] = @benchmarkable chollogdet($B)
        SUITE[string(T)]["Matrix"]["dim:" * string(dim)]["cholinv_logdet"] = @benchmarkable cholinv_logdet($B)

        SUITE[string(T)]["Diagonal"]["dim:" * string(dim)] = BenchmarkGroup()

        # Diagonal{T} benchmarks
        SUITE[string(T)]["Diagonal"]["dim:" * string(dim)]["fastcholesky"]= @benchmarkable fastcholesky($(Diagonal(ones(T, dim))))
        SUITE[string(T)]["Diagonal"]["dim:" * string(dim)]["cholinv"] = @benchmarkable cholinv($(Diagonal(ones(T, dim))))
        SUITE[string(T)]["Diagonal"]["dim:" * string(dim)]["cholsqrt"] = @benchmarkable cholsqrt($(Diagonal(ones(T, dim))))
        SUITE[string(T)]["Diagonal"]["dim:" * string(dim)]["chollogdet"] = @benchmarkable chollogdet($(Diagonal(ones(T, dim))))
        SUITE[string(T)]["Diagonal"]["dim:" * string(dim)]["cholinv_logdet"] = @benchmarkable cholinv_logdet($(Diagonal(ones(T, dim))))
    end
end

SUITE["UniformScaling"] = BenchmarkGroup()

# Uniformscaling benchmarks
SUITE["UniformScaling"]["fastcholesky"] = @benchmarkable fastcholesky($(3.0I))
SUITE["UniformScaling"]["fastcholesky!"] = @benchmarkable fastcholesky!($(3.0I))
SUITE["UniformScaling"]["cholinv"] = @benchmarkable cholinv($(3.0I))
SUITE["UniformScaling"]["cholsqrt"] = @benchmarkable cholsqrt($(3.0I))
