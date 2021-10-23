from collections import defaultdict
from .functions import (
    Ackley, DixonPrice, Griewank,
    Infinity, Mishra11, Multimodal,
    Plateau, Qing, Quintic,
    Rastringin, Rosenbrock, Schwefel21,
    Schwefel22, Step, StyblinskiTang,
    Trid, Sphere
)
import torch
import os
from tqdm import tqdm

class Benchmark(object):
    def __init__(self, n=2) -> None:
        super().__init__()
        self.n = n
        function_args = dict(
            n=n
        )
        self.target_functions = dict(
            Ackley=Ackley(**function_args),
            DixonPrice=DixonPrice(**function_args),
            Griewank=Griewank(**function_args),
            Infinity=Infinity(**function_args),
            Mishra11=Mishra11(**function_args),
            Multimodal=Multimodal(**function_args),
            Plateau=Plateau(**function_args),
            Qing=Qing(**function_args),
            Quintic=Quintic(**function_args),
            Rastringin=Rastringin(**function_args),
            Rosenbrock=Rosenbrock(**function_args),
            Schwefel21=Schwefel21(**function_args),
            Schwefel22=Schwefel22(**function_args),
            Step=Step(**function_args),
            StyblinskiTang=StyblinskiTang(**function_args),
            Trid=Trid(**function_args),
            Sphere=Sphere(**function_args)
        )
        self.results = {
            target_name : defaultdict(list)
            for target_name in self.target_functions
        }
    
    def run(self, algorithms : dict, algorithm_args : dict = None):
        if algorithm_args is None:
            algorithm_args = dict()
        for target in tqdm(self.target_functions):
            for algo in tqdm(algorithms):
                objective, compute_time,\
                solution, visited_points = algorithms[algo](
                    objective=self.target_functions[target],
                    **algorithm_args[algo]
                    )
                self.results[target][algo].append((objective, compute_time, solution, visited_points))
    
    def summary(self, label="objective", root_dir="./result/picture"):
        """
        共通：計算時間順にソート、目的関数値順にソート、アルゴリズムごとの統計情報算出
        低次元：結果の可視化
        """
        os.makedirs(root_dir, exist_ok=True)
        for target in self.target_functions:
            print(target)
            for algo in self.results[target]:
                for i, result in enumerate(self.results[target][algo]):
                    print(algo, result[0])
                    if self.n == 2:
                        points = torch.stack(result[-1])
                        self.target_functions[target].heatmap(points=points, gif_title=f"{root_dir}/heatmap_{target}_{algo}_{i}.gif")
                        self.target_functions[target].plot2D(points=points, gif_title=f"{root_dir}/3Dplot_{target}_{algo}_{i}.gif")
                if label == "objective":
                    print(algo, self.results[target][algo][0][0])
                elif label == "2Dplot" and self.n == 2:
                    points = torch.stack(self.results[target][algo][0][-1])
                    self.target_functions[target].heatmap(points=points, gif_title=f"{root_dir}/heatmap_{target}_{algo}_{i}.gif")
                    self.target_functions[target].plot2D(points=points, gif_title=f"{root_dir}/3Dplot_{target}_{algo}_{i}.gif")
                elif label == "3Dheatmap" and self.n == 3:
                    points = torch.stack(self.results[target][algo][0][-1])
                    self.target_functions[target].heatmap3D(points=points, gif_title=f"{root_dir}/heatmap3D_{target}_{algo}.gif")
                else:
                    raise NotImplementedError

    def reset(self):
        self.results = {
            target_name : defaultdict(list)
            for target_name in self.target_functions
        }