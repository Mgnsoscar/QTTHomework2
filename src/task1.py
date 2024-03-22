import matplotlib.pyplot as plt
import random
import numpy as np
import json
import os
from typing import List, Optional, Tuple, Dict
from scipy.constants import elementary_charge as e_ch
from copy import copy

    
class Task1:
    
    C_L: float
    C_g1: float
    C_12: float
    C_1: float
    
    def __init__(self) -> None:
        
        self.C_L = 0.2*np.abs(e_ch) * 1e3
        self.C_g1 = 0.3*np.abs(e_ch) * 1e3
        self.C_12 = 0
        self.C_1 = self.C_L + self.C_g1 + self.C_12
        self.C_2 = self.C_L + self.C_g1 + self.C_12
    
    def V_g2(self, V_g1: np.ndarray, N_1: int, N_2: int) -> np.ndarray:
        

        numerator = (V_g1 * 1e3 * (1 - N_1)) + N_1**2 + N_2**2 - 2
        denominator = (N_2 - 1) * 1e3
        
        return numerator / denominator
    
    def plot(self) -> None:
        
        V_g1 = np.linspace(0, 10, 100)
        
        # Plot the value in the x-y plane
        plt.figure(figsize=(8, 6))
        
        N1s = [0, 1, 2, 3]
        
        for N_1 in N1s:
            
            for N_2 in N1s:
                
                if N_2 != 1:
                    plt.plot(V_g1, self.V_g2(V_g1, N_1, N_2))
                            
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Value plot in x-y plane')
        plt.grid(True)
        plt.show()
        
if __name__ == "__main__":
    
    task1 = Task1()
    task1.plot()