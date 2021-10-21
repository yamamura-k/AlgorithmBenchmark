from collections import defaultdict
from .functions import (
    Ackley, DixonPrice, Griewank,
    Infinity, Mishra11, Multimodal,
    Plateau, Qing, Quintic,
    Rastringin, Rosenbrock, Schwefel21,
    Schwefel22, Step, StyblinskiTang,
    Trid, Sphere
)

class Benchmark(object):
    def __init__(self, n=2) -> None:
        super().__init__()
        function_args = dict(
            n=n
        )
        self.target_functions = dict(
            # Ackley=Ackley(**function_args),
            # DixonPrice=DixonPrice(**function_args),
            # Griewank=Griewank(**function_args),
            # Infinity=Infinity(**function_args),
            # Mishra11=Mishra11(**function_args),
            # Multimodal=Multimodal(**function_args),
            # Plateau=Plateau(**function_args),
            # Qing=Qing(**function_args),
            # Quintic=Quintic(**function_args),
            # Rastringin=Rastringin(**function_args),
            # Rosenbrock=Rosenbrock(**function_args),
            # Schwefel21=Schwefel21(**function_args),
            # Schwefel22=Schwefel22(**function_args),
            # Step=Step(**function_args),
            # StyblinskiTang=StyblinskiTang(**function_args),
            # Trid=Trid(**function_args),
            Sphere=Sphere(**function_args)
        )
        self.results = {
            target_name : defaultdict(list)
            for target_name in self.target_functions
        }
    
    def run(self, algorithms : dict, algorithm_args : dict = None):
        if algorithm_args is None:
            algorithm_args = dict()
        for target in self.target_functions:
            for algo in algorithms:
                objective, compute_time,\
                solution, visited_points = algorithms[algo](
                    objective=self.target_functions[target],
                    **algorithm_args[algo]
                    )
                self.results[target][algo].append((objective, compute_time, solution, visited_points))
    
    def summary(self, label="objective"):
        """
        共通：計算時間順にソート、目的関数値順にソート、アルゴリズムごとの統計情報算出
        低次元：結果の可視化
        """
        for target in self.target_functions:
            print(target)
            for algo in self.results[target]:
                if label == "objective":
                    print(algo, self.results[target][algo][0][0].item())
                else:
                    raise NotImplementedError
