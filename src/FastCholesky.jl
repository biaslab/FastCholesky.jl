module FastCholesky

using LinearAlgebra
using PositiveFactorizations

import LinearAlgebra: BlasInt, BlasFloat

export fastcholesky, fastcholesky!, cholinv, cholsqrt, chollogdet, cholinv_logdet

"""
    fastcholesky(input)

Returns the `Cholesky` factorization of the `input` in the same format as `LinearAlgebra.cholesky`.
By default fallbacks to `LinearAlgebra.cholesky(PositiveFactorizations.Positive, input)`, thus does not requires the input to be perfectly symmetric. 
Provides more efficient implementations for certain inputs.

```jldoctest 
julia> C = fastcholesky([ 1.0 0.5; 0.5 1.0 ]);

julia> C.L * C.L' ≈ [ 1.0 0.5; 0.5 1.0 ]
true
```
"""
function fastcholesky(input::AbstractMatrix)
    return cholesky(PositiveFactorizations.Positive, Hermitian(input))
end
fastcholesky(input::Diagonal) = cholesky(input)
fastcholesky(input::Hermitian) = cholesky(PositiveFactorizations.Positive, input)
fastcholesky(input::Number) = input

function fastcholesky(x::UniformScaling)
    return error(
        "`fastcholesky` is not defined for `UniformScaling`. The shape is not determined."
    )
end
function fastcholesky!(x::UniformScaling)
    return error(
        "`fastcholesky!` is not defined for `UniformScaling`. The shape is not determined."
    )
end

function fastcholesky(input::Matrix{<:BlasFloat})
    C = fastcholesky!(copy(input))
    return isposdef(C) ? C : cholesky(PositiveFactorizations.Positive, Hermitian(input))
end

"""
    fastcholesky!(input)

In-place version of the `fastcholesky`. Does not check the positive-definiteness of the `input` and does not throw. 
Use `LinearAlgebra.issuccess` to check if the result is a proper Cholesky factorization.

```jldoctest 
julia> C = fastcholesky([ 1.0 0.5; 0.5 1.0 ]);

julia> C.L * C.L' ≈ [ 1.0 0.5; 0.5 1.0 ]
true
```
"""
function fastcholesky! end

function fastcholesky!(A::Matrix{<:BlasFloat})
    n = LinearAlgebra.checksquare(A)
    # Heuristics, the built-in version is faster for large inputs
    # Perhaps due to the better cache usage, the Base version fallbacks to LAPACK
    return n < 100 ? _fastcholesky!(n, A) : cholesky!(A; check=false)
end

function _fastcholesky!(n, A::AbstractMatrix)
    @inbounds @fastmath for col in 1:n
        @simd for idx in 1:(col - 1)
            A[col, col] -= A[col, idx]^2
        end
        if A[col, col] <= 0
            return Cholesky(A, 'L', convert(BlasInt, -1))
        end
        A[col, col] = sqrt(A[col, col])
        invAcc = inv(A[col, col])
        for row in (col + 1):n
            @simd for idx in 1:(col - 1)
                A[row, col] -= A[row, idx] * A[col, idx]
            end
            A[row, col] *= invAcc
        end
    end
    return Cholesky(A, 'L', convert(BlasInt, 0))
end

"""
    cholinv(input)

Returns an inverse of the `input`, but uses the Cholesky factorization.
An alias for `inv(fastcholesky(input))`. 
"""
cholinv(input::AbstractMatrix) = inv(fastcholesky(input))

cholinv(input::UniformScaling) = inv(input.λ) * I
cholinv(input::Diagonal) = inv(input)
cholinv(input::Number) = inv(input)

function cholinv(input::Matrix{<:AbstractFloat})
    C = fastcholesky(input)
    LinearAlgebra.inv!(C)
    return C.factors
end

"""
    cholsqrt(input)

Returns an squared root of the matrix `input`, but uses the Cholesky factorization.
An alias for `fastcholesky(x).L`.
"""
cholsqrt(input) = fastcholesky(input).L

cholsqrt(input::UniformScaling) = sqrt(input.λ) * I
cholsqrt(input::Diagonal) = Diagonal(sqrt.(diag(input)))
cholsqrt(input::Number) = sqrt(input)

"""
    cholsqrt(input)

Returns a log-determinant of the matrix `input`, but uses the Cholesky factorization.
An alias for `logdet(fastcholesky(input))`.
"""
chollogdet(input) = logdet(fastcholesky(input))

chollogdet(input::UniformScaling) = logdet(input)
chollogdet(input::Diagonal) = logdet(input)
chollogdet(input::Number) = logdet(input)

"""
    cholinv_logdet(input)

Returns an inverse and a log-determinant of the matrix `input` simultaneously, but uses the Cholesky factorization.
"""
function cholinv_logdet(input)
    C = fastcholesky(input)
    return inv(C), logdet(C)
end

function cholinv_logdet(input::Matrix{<:AbstractFloat})
    C = fastcholesky(input)
    lC = logdet(C)
    LinearAlgebra.inv!(C)
    return C.factors, lC
end

cholinv_logdet(input::UniformScaling) = inv(input), logdet(input)
cholinv_logdet(input::Diagonal) = inv(input), logdet(input)
cholinv_logdet(input::Number) = inv(input), log(abs(input))

end
