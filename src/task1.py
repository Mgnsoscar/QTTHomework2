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

            
    def V_g2(self, N1: int, N2: int, Vg1: np.ndarray) -> np.ndarray:
        
        numerator = (
            e_ch*(
            (2*(N1 - 1)*((self.C_g1*Vg1/e_ch))) - 
            (N1**2 + N2**2) +
            2
            )
        )
        denominator = (
            2 * (1-N2) * self.C_g2
        )

        return numerator / denominator

    def V_g1(self, N1: int, N2: int, Vg2: np.ndarray) -> np.ndarray:
        
        numerator = (
            e_ch*(
                (2*(N2 - 1)*((self.C_g2*Vg2)/e_ch)) - 
                (N1**2 + N2**2) +
                2
            )
        )
            
        denominator = (
            2 * (1-N1) * self.C_g1
        )

        return numerator / denominator

    def V_g2_task_b(self, N1:int, N2:int, Vg2: np.ndarray) -> np.ndarray:
        
        
        
        numerator =(
            -e_ch * (
                self.C_1 * (
                    (2*((self.C_g2*Vg2)/e_ch)) + (N1**2+N2**2) - 2
                )
                +
                self.C_L * (
                    (-2*((self.C_g2*Vg2)/e_ch)*(N1 - 1)) + (N1 * N2) - 1
                )
            )
        )
        denominator = (
            2 * (( (N1 - 1)*self.C_1 ) + ((N2 - 1) * self.C_L ) ) * self.C_g1
        )
        
        return numerator / denominator
        
    def plot_a(self) -> None:
        """
        Plot V_g1 vs V_g2 for various (N_1, N_2) configurations.
        """

        # Generate values for V_g1 and V_g2
        V_g1 = np.linspace(0, 1e-2, 100)
        V_g2 = np.linspace(0, 1e-2, 100)

        # Create a figure
        plt.figure(figsize=(8, 6))

        # Define configurations of (N_1, N_2)
        N1s = [0, 1, 2]

        # Plot V_g1 vs V_g2 for each (N_1, N_2) configuration
        for N_1 in N1s:
            for N_2 in N1s:
                Vg2 = self.V_g2(N_1, N_2, V_g1)
                Vg1 = self.V_g1(N_1, N_2, Vg2)

                # Check for valid values and plot
                if not np.isnan(Vg2[0]):
                    plt.plot(V_g2, Vg1, label=f"$(N_1,N_2) = ({N_1},{N_2})$")

        # Plot specific configurations
        two_one = self.V_g1(2, 1, V_g1)
        two_zero = self.V_g1(0, 1, V_g1)
        plt.plot(two_one, np.linspace(-0.01, 0.01, 100), label="$(N_1,N_2) = (2,1)$")
        plt.plot(two_zero, np.linspace(-0.01, 0.01, 100), label="$(N_1,N_2) = (0,1)$")

        # Set plot limits and labels
        plt.xlim(0, 0.010)
        plt.ylim(-0.01, 0.01)
        plt.xlabel('$V_{g1} [V]$')
        plt.ylabel('$V_{g2} [V]$')
        plt.title('$U(N_1,N_2)=U(1,1)$')
        plt.legend()
        plt.grid(False)
        plt.show()

        
    def plot_b(self) -> None:
        """
        Plot V_g1 vs V_g2 for various (N_1, N_2) configurations.
        """

        # Generate values for V_g1 and V_g2
        V_g1 = np.linspace(-0.01, 10e-3, 100)
        V_g2 = np.linspace(-0.01, 10e-3, 100)

        # Create a figure
        plt.figure(figsize=(8, 6))

        # Define configurations of (N_1, N_2)
        N1s = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)]

        # Plot V_g1 vs V_g2 for each (N_1, N_2) configuration
        for Ns in N1s:
            
            Vg1 = self.V_g2_task_b(Ns[0], Ns[1], V_g2)

            # Check for valid values and plot
            if not np.isnan(Vg1[0]):
                plt.plot(V_g2, Vg1, label=f"$(N_1,N_2) = ({Ns[0]},{Ns[1]})$")


        # Set plot limits and labels
        plt.xlim(-0.01, 0.010)
        plt.ylim(-0.01, 0.01)
        plt.xlabel('$V_{g2} [V]$')
        plt.ylabel('$V_{g1} [V]$')
        plt.title('$U(N_1,N_2)=U(1,1)$')
        plt.legend()
        plt.grid(False)
        plt.show()
        
if __name__ == "__main__":
    
    task1 = Task1()
    task1.plot_b()