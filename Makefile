SHELL = /bin/bash
.DEFAULT_GOAL = help


.PHONY: test

test: ## Run tests
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test();'

.PHONY: benchmark

benchmark: ## Run benchmarks
	julia -e '\
    import Pkg; \
    Pkg.activate("benchmarks/speed"); \
    Pkg.develop(Pkg.PackageSpec(path=pwd())); \
    Pkg.instantiate(); \
    using PkgBenchmark, FastCholesky, Dates; \
    results = benchmarkpkg("FastCholesky"; script="benchmarks/speed/benchmarks.jl"); \
    export_markdown("benchmarks/speed/exports/benchmark-"*Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM")*".md", results); \
    Base.Filesystem.rm("benchmark"; force=true, recursive=true)'