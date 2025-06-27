#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 18:38:26 2025

Jorge Salas C17105
"""

# -*- coding: utf-8 -*-
"""
Simulación EFM para muestras dieléctricas hemisféricas
Basado en: Gómez et al. (2010) - Estimación de carga superficial mediante EFM

"""


import numpy as np
import scipy as sp
from scipy.special import eval_legendre as legendre  # Polinomios de Legendre
import matplotlib.pyplot as plt



epsilon_0 = 8.854187e-12  # Permitividad del vacío [F/m]

# Geometría de la punta AFM (Figura 1 del artículo)
h_T = 0.5e-6    # Altura de la punta [m]
R = 20e-9       # Radio del ápice [m]
alpha = np.deg2rad(20)  # Ángulo semi-apertura [rad]

# Parámetros de la muestra
a = 1e-6        # Radio del hemisferio [m]
k_e = 3.0       # Constante dieléctrica relativa
sigma0 = 0.100e-6  # Densidad de carga superficial inicial [C/m²]

# Configuración experimental
d0 = 75e-9      # Separación punta-muestra [m]
V0 = 10.0       # Voltaje de polarización [V]

# Parámetros numéricos
N_anillos = 30      # Número de anillos de carga
N_theta = 30        # Puntos de discretización angular
N_terminos = 30     # Truncamiento de serie de Legendre
P_z = 0.100e-6

# Calculo de los potenciales

def potencial_hemisferio(r, theta):
    """
    Calcula el potencial creado por el hemisferio cargado (Ecuación 4 del artículo)
    
    """
    suma = 0.0
    
    # Término n=1 (calculado aparte por eficiencia)
    I_1 = legendre(0, 0) - legendre(2, 0)
    termino = (1/(k_e + 2)) * (a/r) * (1/a) * legendre(1, np.sin(theta)) * I_1
    suma += termino
    
    # Suma para n >= 2
    for n in range(2, N_terminos + 1):
        I_n = legendre(n-1, 0) - legendre(n+1, 0)
        P_n = legendre(n, np.sin(theta))
        termino = (1/(k_e*n + n + 1)) * (a/r)**n * (1/a) * P_n * I_n
        suma += termino
        
    return 2 * np.pi * (a**2) * sigma0 * suma


def campo_electrico_radial(theta):
    """
    Calcula la componente radial del campo eléctrico
    
    """
    return P_z * (np.sin(theta) / (epsilon_0 * (k_e + 2)))

def densidad_carga_total(theta, d, V):
    """
    Calcula la densidad de carga superficial total
    
    Incluye:
    - Polarización inducida (término k_e-1)
    - Carga inicial (sigma0)
    - Interacción con la punta (término P_z)
    """
    campo = campo_electrico_radial(theta)
    return (k_e - 1) * epsilon_0 * campo + sigma0 + P_z * np.sin(theta)

def integrando(theta, d, V):
    """Función integrando para cálculo de carga total"""
    return densidad_carga_total(theta, d, V) * np.sin(theta) * a**2


if __name__ == "__main__":
    # Configuración de ángulos
    theta = np.linspace(0, np.pi/2, N_theta)
    campo_radial = np.zeros_like(theta)
    
    # Prefactor común
    factor = 2 * np.pi * a**2 * sigma0 / epsilon_0
    
    # Cálculo del campo radial en cada punto
    for i, angulo in enumerate(theta):
        suma_serie = 0.0
        
        for n in range(1, N_terminos + 1):
            I_n = legendre(n-1, 0) - legendre(n+1, 0)
            P_n = legendre(n, np.sin(angulo))
            coef = n * (1/(k_e*n + n + 1)) * (a/a)**(n-1) * I_n
            suma_serie += coef * P_n
            
        campo_radial[i] = -factor * suma_serie
    
    print(f"Campo eléctrico máximo: {np.max(campo_radial):.2e} V/m")
    print(f"Campo eléctrico mínimo: {np.min(campo_radial):.2e} V/m")
