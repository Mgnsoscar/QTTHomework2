import matplotlib.pyplot as plt
import random
import numpy as np
import json
import os
from typing import List, Optional, Tuple, Dict
from scipy.constants import elementary_charge as e_ch
from copy import copy
from sympy import symbols, Eq, solve
    
class Task1:
    
    C_L: float
    C_g1: float
    C_12: float
    C_1: float
    
    def __init__(self) -> None:
        
        self.C_L = 0.2*e_ch*1e3
        self.C_g1 = 0.3*e_ch*1e3
        self.C_g2 = 0.3*e_ch*1e3
        self.C_12 = 0
        self.C_1 = self.C_L + self.C_g1 + self.C_12
        self.C_2 = self.C_L + self.C_g1 + self.C_12
    
    def do_task_a(self):
        self.stability_diagram_task_a(
            function_for_Vg1 = self.V_g1_task_a,
            function_for_Vg2 = self.V_g2_task_a
        )
    
    def do_task_b(self) -> None:
        self.stability_diagram_task_b(
            function_for_Vg1 = self.V_g1_task_b,
            function_for_Vg2 = self.V_g2_task_b
        )
            
    def V_g2_task_a(self, x: int, y: int, N1: int, N2: int, Vg1: np.ndarray) -> np.ndarray:
        
        numerator = e_ch * (
            (y**2 - N2**2) - (N1**2 - x**2) + ((2 * self.C_g1 * Vg1 * (N1 - x))/e_ch)
        )
        denominator = (
            2 * (y - N2) * self.C_g1
        )

        return numerator / denominator

    def V_g1_task_a(self, x: int, y: int, N1: int, N2: int, Vg2: np.ndarray) -> np.ndarray:
        
        numerator = e_ch * (
            (N1**2 - x**2) - (y**2 - N2**2) + ((2 *self.C_g2 * Vg2 * (y - N2))/e_ch)
        )
        denominator = (
            2 * (N1 - x) * self.C_g1
        )

        try:
            return numerator / denominator
        except:
            return False
        
        return numerator / denominator     

    def V_g2_task_b(self, x: int, y:int, N1:int, N2:int, Vg1: np.ndarray) -> np.ndarray:

        c12 = self.C_L
        cg1 = self.C_g1
        cg2 = self.C_g2
        c2 = self.C_L + c12 + cg2
        
        numerator = (e_ch / (2*cg2)) * (
            2*c12*( N1*N2 + cg1*Vg1*(1/e_ch)*(y-N2) - y ) + c2*(N1**2 + N2**2 - (x**2 + y**2) - (2/e_ch)*cg1*Vg1*(N1-x))
        )
        
        denominator = (
            c2*(N2-y) + c12*(N1 - x)
        )
        
        return numerator / denominator
    
    def V_g1_task_b(self, x: int, y:int, N1:int, N2:int, Vg2: np.ndarray) -> np.ndarray:
        
        c12 = self.C_L
        cg1 = self.C_g1
        cg2 = self.C_g2
        c2 = self.C_L + c12 + cg1
        
        numerator = (e_ch / (2*cg1)) * (
            2*c12*( N1*N2 + cg2*Vg2*(1/e_ch)*(x-N1) - y ) + c2*(N1**2 + N2**2 - (x**2 + y**2) - (2/e_ch)*cg2*Vg2*(N2-y))
        )
        
        denominator = (
            c2*(N1-x) + c12*(N2 - y)
        )
        
        return numerator / denominator
    
        
    def stability_diagram_task_a(self, function_for_Vg1, function_for_Vg2) -> None:
        """
        Plot V_g1 vs V_g2 for various (N_1, N_2) configurations.
        """

        # Generate values for V_g1 and V_g2
        Vg1 = np.linspace(0, 1e-2, 100)
        Vg2 = np.linspace(0, 1e-2, 100)

        # Create a figure
        plt.figure(figsize=(8, 6))
        
        # Define configurations of (N_1, N_2)

        N1 = 0
        N2 = 1
        for x in range(1, 4):

            Vg1_function = function_for_Vg1(x, x, N1, N2, Vg2)
            Vg2_function = function_for_Vg2(x, x, N2, N1, Vg1)
                        
            # Plot Vg2 as function of Vg1
            plt.plot(Vg2, Vg1_function, label=f"$U({N1},{N2}) = U({x},{x})$")
            plt.plot(Vg2_function, Vg1, label=f"$U({N2},{N1}) = U({x}, {x})$")
                
            N1 += 1
            N2 += 1
                
        plt.xlabel('$V_{g1} [V]$')
        plt.ylabel('$V_{g2} [V]$')
        
        plt.title('$U(N_1,N_2)=U(N_x,N_y)$')
        
        plt.legend()
        plt.grid(False)
        plt.show()

    def stability_diagram_task_b(self, function_for_Vg1, function_for_Vg2) -> None:
        """
        Plot V_g1 vs V_g2 for various (N_1, N_2) configurations.
        """

        # Generate values for V_g1 and V_g2
        Vg1 = np.linspace(0, 1e-2, 100)
        Vg2 = np.linspace(0, 1e-2, 100)

        # Create a figure
        plt.figure(figsize=(8, 6))
        
        # Define configurations of (N_1, N_2)

        N1 = 0
        N2 = 0
        for x in range(1, 2):
            

            for i in range(0, 3):
                    
                for j in range(0, 3):
                    
                    if i != j:
                        try:
                            Vg1_function = function_for_Vg1(x, x, N1 + i, N2 + j, Vg2)
                            plt.plot(Vg2, Vg1_function, label=f"$U({N1 + i},{N2 + j}) = U({x},{x})$")
                        except:
                            pass
                        #try: 
                        #    Vg2_function = function_for_Vg2(x, x, N2 + j, N1 + i, Vg1)
                        #    plt.plot(Vg2_function, Vg1, label=f"$({N2 + j},{N1 + i}) = ({x}, {x})$")
                        #except:
                        #    pass
                    
                    
            N1 += 1
            N2 += 1
                
        plt.xlabel('$V_{g1} [V]$')
        plt.ylabel('$V_{g2} [V]$')
        
        plt.title('$U(N_1,N_2)=U(N_x,N_y)$')
        
        plt.xlim(0, 0.01)
        plt.ylim(0, 0.01)
        plt.legend()
        plt.grid(False)
        plt.show()
        
if __name__ == "__main__":
    
    task1 = Task1()
    task1.do_task_b()