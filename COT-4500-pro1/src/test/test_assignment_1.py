import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.main.assignment_1 import approx_alg
from src.main.assignment_1 import bisection_method
from src.main.assignment_1 import fixed_point
from src.main.assignment_1 import newtons_method
import numpy as np

def g1(x):
    result = x - x**3 - 4*x**2 + 10  # Matches Java code exactly

        # Stop early if result becomes too large
    if abs(result) > 1e10:  # Java diverges after ~10^8, so we set a threshold
        return float("nan")  # Return NaN to indicate divergence
        
    return result

def g2(x):
    """ Function (b) - Expected to converge """
    expr = (10 - x**3)  # Compute inside the square root
    if expr < 0:
        return float("nan")  # Avoid taking sqrt of negative numbers
    return np.sqrt(expr) / 2  # Corrected formula

def test_sqrt_2():
    result = approx_alg(np.sqrt(2))
    expected = np.sqrt(2)

    print(f"Calculated result: {result}")
    print(f"Expected result: {expected}")

    assert abs(result - expected) < 1e-5  # Check tolerance

def test_bisection_root_finding():       
        def f(x):
            return x**3 + 4*x**2 - 10    # Function with a known root near 1.521

        a, b = 1, 2  # Interval where f(x) changes sign
        tol = 0.001
        
        result = bisection_method(f, a, b, tol)
        expected = 1.365230560302  # Precomputed correct root
        
        print(f"Calculated result: {result}")
        print(f"Expected result: {expected}")
        

def test_fixed_point_iteration():
    p0 = 1.5  # Initial guess (same as Java code)
    tol = 1e-6  # Tolerance (same as Java)
    max_iter = 50  # Maximum iterations (same as Java)

    # Run fixed-point iteration for both functions
    print("\nRunning Fixed-Point Iteration for g1(x) (Expected to Diverge):")
    root1, iter1, status1, results1 = fixed_point(g1, p0, tol, max_iter)

    print("\nRunning Fixed-Point Iteration for g2(x) (Expected to Converge):")
    root2, iter2, status2, results2 = fixed_point(g2, p0, tol, max_iter)

    # Print formatted results for g1
    print("\nIterations for g1(x) (Expected to Diverge):")
    for iteration, value in results1:
        print(f"{iteration} : {value:.15f}")

    # Print formatted results for g2
    print("\nIterations for g2(x) (Expected to Converge):")
    for iteration, value in results2:
        print(f"{iteration} : {value:.15f}")

    print(f"\nFinal Result for g1(x): {root1:.15f}, Iterations: {iter1}, Status: {status1}")
    print(f"Final Result for g2(x): {root2:.15f}, Iterations: {iter2}, Status: {status2}")




def test_newtons_method():
    # Define f(x) and f'(x)
    def f(x):
        return np.cos(x) - x

    def df(x):
        return -np.sin(x) - 1  # Correct derivative

    
    p0 = np.pi / 4  # Initial guess from example
    tol = 1e-10  # Precision tolerance
    max_iter = 10  # Sufficient for convergence

    # Run Newton's Method
    root, iter_count, status, results = newtons_method(f, df, p0, tol, max_iter)

    # Print formatted results
    print("\nIterations for Newton's Method:")
    for iteration, value in results:
        print(f"{iteration} : {value:.10f}")

    print(f"\nFinal Result: {root:.10f}, Iterations: {iter_count}, Status: {status}")



        

if __name__ == "__main__":
    
    print("Approximation Algorithm Test")
    test_sqrt_2()

    print("\n\nBisection Method Algorithm Test")
    test_bisection_root_finding()

    print("\n\nFixed Point Algorithm Test")
    test_fixed_point_iteration()

    print("\n\nNewton Method Algorithm Test")
    test_newtons_method()