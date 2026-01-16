# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n <= 0:
        return None

    k = np.arange(n)
    nodes = np.cos(np.pi * k / (n - 1))
    return nodes
    


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n <= 0:
        return None

    w = np.ones(n)
    if n == 1:
        return w

    w[0] = 0.5        
    w[-1] = -0.5      

    if n > 2:
        w[1:-1] = (-1) ** np.arange(1, n-1)  

    return w


def f_1(x):
    return np.sign(x)*x + x**2

def f_2(x):
    return np.sign(x)*x**2

def f_3(x):
    return (abs(math.sin(5*x)))**3

def f_4_1(x):
    return 1/(1 + 1 * x**2)

def f_4_25(x):
    return 1/(1 + 25 * x**2)

def f_4_100(x):
    return 1/(1 + 100 * x**2)

def f_5(x):
    return np.sign(x)

def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not all(isinstance(arr, np.ndarray) for arr in (xi, yi, wi, x)):
        return None
    if xi.ndim != 1 or yi.ndim != 1 or wi.ndim != 1 or x.ndim != 1:
        return None
    if not (len(xi) == len(yi) == len(wi)):
        return None

    n = len(x)
    result = np.zeros(n)
    
    for i in range(n):
        diff = x[i] - xi
        
        
        mask = np.abs(diff) < 1e-12
        if np.any(mask):
            result[i] = yi[mask][0]
        else:
            numerator = np.sum(wi * yi / diff)
            denominator = np.sum(wi / diff)
            result[i] = numerator / denominator
    
    return result


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        xr_arr = np.asarray(xr, dtype=float)
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None

    if xr_arr.shape != x_arr.shape:
        return None

    return float(np.max(np.abs(xr_arr - x_arr)))

