var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = FastCholesky","category":"page"},{"location":"#FastCholesky","page":"Home","title":"FastCholesky","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package exports fastcholesky function, which works exactly like the cholesky from the Julia, but faster!","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [FastCholesky]","category":"page"},{"location":"#FastCholesky.cholinv-Tuple{AbstractMatrix}","page":"Home","title":"FastCholesky.cholinv","text":"cholinv(input)\n\nReturns an inverse of the input, but uses the Cholesky factorization. An alias for inv(fastcholesky(input)). \n\n\n\n\n\n","category":"method"},{"location":"#FastCholesky.cholinv_logdet-Tuple{Any}","page":"Home","title":"FastCholesky.cholinv_logdet","text":"cholinv_logdet(input)\n\nReturns an inverse and a log-determinant of the matrix input simultaneously, but uses the Cholesky factorization.\n\n\n\n\n\n","category":"method"},{"location":"#FastCholesky.chollogdet-Tuple{Any}","page":"Home","title":"FastCholesky.chollogdet","text":"cholsqrt(input)\n\nReturns a log-determinant of the matrix input, but uses the Cholesky factorization. An alias for logdet(fastcholesky(input)).\n\n\n\n\n\n","category":"method"},{"location":"#FastCholesky.cholsqrt-Tuple{Any}","page":"Home","title":"FastCholesky.cholsqrt","text":"cholsqrt(input)\n\nReturns an squared root of the matrix input, but uses the Cholesky factorization. An alias for fastcholesky(x).L.\n\n\n\n\n\n","category":"method"},{"location":"#FastCholesky.fastcholesky!","page":"Home","title":"FastCholesky.fastcholesky!","text":"fastcholesky!(input)\n\nIn-place version of the fastcholesky. Does not check the positive-definiteness of the input and does not throw.  Use LinearAlgebra.issuccess to check if the result is a proper Cholesky factorization.\n\njulia> fastcholesky!([ 1.0 0.5; 0.5 1.0 ])\nCholesky{Float64, Matrix{Float64}}\nL factor:\n2×2 LowerTriangular{Float64, Matrix{Float64}}:\n 1.0   ⋅ \n 0.5  0.866025\n\n\n\n\n\n","category":"function"},{"location":"#FastCholesky.fastcholesky-Tuple{AbstractMatrix}","page":"Home","title":"FastCholesky.fastcholesky","text":"fastcholesky(input)\n\nReturns the Cholesky factorization of the input in the same format as LinearAlgebra.cholesky. By default fallbacks to LinearAlgebra.cholesky(PositiveFactorizations.Positive, input), thus does not requires the input to be perfectly symmetric.  Provides more efficient implementations for certain inputs.\n\njulia> fastcholesky([ 1.0 0.5; 0.5 1.0 ])\nCholesky{Float64, Matrix{Float64}}\nL factor:\n2×2 LowerTriangular{Float64, Matrix{Float64}}:\n 1.0   ⋅ \n 0.5  0.866025\n\n\n\n\n\n","category":"method"}]
}
