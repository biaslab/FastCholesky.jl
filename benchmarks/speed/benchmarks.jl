using BenchmarkTools
using FastCholesky

const SUITE = BenchmarkGroup()

for T in (Float32, Float64, BigFloat)

    for dim in (2, 5, 10)#, 20, 50, 100, 200, 500)

        # generate positive-definite matrix of specified dimensions
        A = randn(dim, dim)
        B = A * A'
        C = similar(B)

        # define benchmarks
        SUITE["fastcholesky-"*string(T)*"-"*string(dim)]  = @benchmarkable fastcholesky($B)
        SUITE["fastcholesky!-"*string(T)*"-"*string(dim)] = @benchmarkable fastcholesky!($C)
        SUITE["cholinv-"*string(T)*"-"*string(dim)] = @benchmarkable cholinv($B)
        SUITE["cholsqrt-"*string(T)*"-"*string(dim)] = @benchmarkable cholsqrt($B)
        SUITE["chollogdet-"*string(T)*"-"*string(dim)] = @benchmarkable chollogdet($B)
        SUITE["cholinv_logdet-"*string(T)*"-"*string(dim)] = @benchmarkable cholinv_logdet($B)
    
    end

end