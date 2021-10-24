from benchmarks import Benchmark
from algorithms.metaheuristics import (
    ABC, BA, FA, GWO, TLBO
    )
from algorithms.grad_based import (
    GradientDescent, ConjugateGradientDescent,
    NesterovAcceralation
)
from projection import BoxProjection
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
    n = 20
    benchmark = Benchmark(n=n)
    algorithm_arg = dict(
        dimension=n,
        max_iter=30,
        method="wolfe",
        proj=BoxProjection(1, 0),
    )
    beta_methods = ["FR","PR","HS","DY","HZ","DL","LS"]
    algorithm_args = {algo : algorithm_arg for algo in algorithms}
    benchmark.run(algorithms, algorithm_args)
    for beta in beta_methods:
        algorithm_args["ConjugateGradientDescent"]["beta_method"] = beta
        conjugate = {"ConjugateGradientDescent" : algorithms["ConjugateGradientDescent"]}
        benchmark.run(conjugate, algorithm_args)

    benchmark.summary(root_dir=f"./result/picture/")
    benchmark.reset()

if __name__=='__main__':
    main()