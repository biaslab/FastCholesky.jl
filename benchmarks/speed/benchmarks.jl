using BenchmarkTools, LinearAlgebra
using FastCholesky

const SUITE = BenchmarkGroup()

# general benchmarks
for T in (Float32, Float64, BigFloat)

    for dim in (2, 5, 10)#, 20, 50, 100, 200, 500)

        # generate positive-definite matrix of specified dimensions
        A = randn(dim, dim)
        B = A * A'
        C = similar(B)

        # define benchmarks
        SUITE["fastcholesky-"  *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable fastcholesky($B)
        SUITE["fastcholesky!-" *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable fastcholesky!($C)
        SUITE["cholinv-"       *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable cholinv($B)
        SUITE["cholsqrt-"      *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable cholsqrt($B)
        SUITE["chollogdet-"    *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable chollogdet($B)
        SUITE["cholinv_logdet-"*"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable cholinv_logdet($B)
    
    end

end

# Float64 benchmarks
SUITE["fastcholesky-"  *"Type:Float64"] = @benchmarkable fastcholesky($3.0)
SUITE["fastcholesky!-" *"Type:Float64"] = @benchmarkable fastcholesky!($3.0)
SUITE["cholinv-"       *"Type:Float64"] = @benchmarkable cholinv($3.0)
SUITE["cholsqrt-"      *"Type:Float64"] = @benchmarkable cholsqrt($3.0)
SUITE["chollogdet-"    *"Type:Float64"] = @benchmarkable chollogdet($3.0)
SUITE["cholinv_logdet-"*"Type:Float64"] = @benchmarkable cholinv_logdet($3.0)

# Diagonal{Float64} benchmarks
SUITE["fastcholesky-"  *"Type:Diagonal{Float64}"] = @benchmarkable fastcholesky($(Diagonal(3.0*ones(100))))
SUITE["cholinv-"       *"Type:Diagonal{Float64}"] = @benchmarkable cholinv($(Diagonal(3.0*ones(100))))
SUITE["cholsqrt-"      *"Type:Diagonal{Float64}"] = @benchmarkable cholsqrt($(Diagonal(3.0*ones(100))))
SUITE["chollogdet-"    *"Type:Diagonal{Float64}"] = @benchmarkable chollogdet($(Diagonal(3.0*ones(100))))
SUITE["cholinv_logdet-"*"Type:Diagonal{Float64}"] = @benchmarkable cholinv_logdet($(Diagonal(3.0*ones(100))))

# Uniformscaling benchmarks
SUITE["fastcholesky-"  *"Type:UniformScaling"] = @benchmarkable fastcholesky($(3.0I))
SUITE["fastcholesky!-" *"Type:UniformScaling"] = @benchmarkable fastcholesky!($(3.0I))
SUITE["cholinv-"       *"Type:UniformScaling"] = @benchmarkable cholinv($(3.0I))
SUITE["cholsqrt-"      *"Type:UniformScaling"] = @benchmarkable cholsqrt($(3.0I))
