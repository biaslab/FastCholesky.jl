using BenchmarkTools, LinearAlgebra
using FastCholesky

const SUITE = BenchmarkGroup()

# general benchmarks
for T in (Float32, Float64, BigFloat)

    # number
    a = rand(T)
    
    SUITE["fastcholesky-"  *"Type:"*string(T)] = @benchmarkable fastcholesky($a)
    SUITE["fastcholesky!-" *"Type:"*string(T)] = @benchmarkable fastcholesky!($a)
    SUITE["cholinv-"       *"Type:"*string(T)] = @benchmarkable cholinv($a)
    SUITE["cholsqrt-"      *"Type:"*string(T)] = @benchmarkable cholsqrt($a)
    SUITE["chollogdet-"    *"Type:"*string(T)] = @benchmarkable chollogdet($a)
    SUITE["cholinv_logdet-"*"Type:"*string(T)] = @benchmarkable cholinv_logdet($a)

    for dim in (2, 5, 10, 20, 50, 100, 200, 500)

        # generate positive-definite matrix of specified dimensions
        A = randn(T, dim, dim)
        B = A * A' + dim * I(dim)
        C = similar(B)

        # define benchmarks
        SUITE["fastcholesky-"  *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable fastcholesky($B)
        SUITE["fastcholesky!-" *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable fastcholesky!($C)
        SUITE["cholinv-"       *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable cholinv($B)
        SUITE["cholsqrt-"      *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable cholsqrt($B)
        SUITE["chollogdet-"    *"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable chollogdet($B)
        SUITE["cholinv_logdet-"*"Type:Matrix{"*string(T)*"}-Dim:"*string(dim)] = @benchmarkable cholinv_logdet($B)
        
        # Diagonal{T} benchmarks
        SUITE["fastcholesky-"  *"Type:Diagonal{"*string(T)*"}"] = @benchmarkable fastcholesky($(Diagonal(ones(T, dim))))
        SUITE["cholinv-"       *"Type:Diagonal{"*string(T)*"}"] = @benchmarkable cholinv($(Diagonal(ones(T, dim))))
        SUITE["cholsqrt-"      *"Type:Diagonal{"*string(T)*"}"] = @benchmarkable cholsqrt($(Diagonal(ones(T, dim))))
        SUITE["chollogdet-"    *"Type:Diagonal{"*string(T)*"}"] = @benchmarkable chollogdet($(Diagonal(ones(T, dim))))
        SUITE["cholinv_logdet-"*"Type:Diagonal{"*string(T)*"}"] = @benchmarkable cholinv_logdet($(Diagonal(ones(T, dim))))
           
    end

end

# Uniformscaling benchmarks
SUITE["fastcholesky-"  *"Type:UniformScaling"] = @benchmarkable fastcholesky($(3.0I))
SUITE["fastcholesky!-" *"Type:UniformScaling"] = @benchmarkable fastcholesky!($(3.0I))
SUITE["cholinv-"       *"Type:UniformScaling"] = @benchmarkable cholinv($(3.0I))
SUITE["cholsqrt-"      *"Type:UniformScaling"] = @benchmarkable cholsqrt($(3.0I))
