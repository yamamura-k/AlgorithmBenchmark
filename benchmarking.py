from benchmarks import Benchmark
from algorithms.metaheuristics import (
    ABC, BA, FA, GWO, TLBO
    )
from algorithms.grad_based import (
    GradientDescent, ConjugateGradientDescent,
    NesterovAcceralation
)
def main():
    algorithms = dict(
        ABC=ABC(),
        BA=BA(),
        FA=FA(),
        GWO=GWO(),
        TLBO=TLBO(),
        GradientDescent=GradientDescent(),
        ConjugateGradientDescent=ConjugateGradientDescent(),
        NesterovAcceralation=NesterovAcceralation(),
        # Newton=Newton()
    )
    n = 2
    benchmark = Benchmark(n=n)
    algorithm_arg = dict(
        dimension=n,
        max_iter=30,
        method="wolfe"
    )
    beta_methods = ["default", "FR","PR","HS","DY","HZ","DL","LS"]
    for beta in beta_methods:
        algorithm_arg["beta_method"] = beta
        algorithm_args = {algo : algorithm_arg for algo in algorithms}
        benchmark.run(algorithms, algorithm_args)
        benchmark.summary(root_dir=f"./result/picture/{beta}")

if __name__=='__main__':
    main()