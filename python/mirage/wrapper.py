import torch
from .core import optimize

class PyGraphWrapper:  
    def __init__(self, graph):  
        self.pygraph = graph
        
        self._is_compiled = False
  
    def __getattr__(self, name):  
        return getattr(self.pygraph, name)
    
    def execute(self, **kwargs):
        if not self._is_compiled:
            self.compile(**kwargs)
            
        
        
        return
    
    def compile(self, **kwargs):
        if self._is_compiled:
            return
        
        graphs = optimize(self.pygraph, **kwargs)
        for i, graph in enumerate(graphs):
            graph.generate_cuda_program("generated_program_{}.cu".format(i))
        # TODO
        
        self._is_compiled = True