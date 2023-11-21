SHELL = /bin/bash
.DEFAULT_GOAL = help

# Windows has different commands in shell
# - `RM` command: use for removing files
ifeq ($(OS), Windows_NT) 
    RM = del /Q /F
    PATH_SEP = \\
else
    RM = rm -rf
    PATH_SEP = /
endif

.PHONY: test

test: ## Run tests, use test_args="folder1:test1 folder2:test2" argument to run reduced testset, use dev=true to use `dev-ed` version of core packages
	julia -e 'ENV["USE_DEV"]="$(dev)"; import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string);'

.PHONY: benchmark

benchmark: ## Run benchmarks
	julia -e '\
    ENV["USE_DEV"]="$(dev)"; \
    import Pkg; \
    Pkg.activate("benchmarks/speed"); \
    using PkgBenchmark, FastCholesky, Dates; \
    results = benchmarkpkg("FastCholesky"; script="benchmarks/speed/benchmarks.jl"); \
    export_markdown("benchmarks/speed/exports/benchmark-"*Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM")*".md", results); \
    Base.Filesystem.rm("benchmark"; force=true, recursive=true)'