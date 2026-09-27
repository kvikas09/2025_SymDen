import sympy as sp
from sympy import sin, cos, Abs

def simplify_and_round_equation(equation, threshold=1e-5, decimals=2):
    """
    Simplifies a symbolic equation, removes terms with coefficients close to zero,
    and rounds the remaining coefficients to a specified number of decimal places.

    Args:
        equation (sympy.Expr): The symbolic equation to simplify.
        threshold (float): The threshold below which coefficients are considered zero.
        decimals (int): The number of decimal places to round coefficients.

    Returns:
        sympy.Expr: The simplified and rounded equation.
    """
    # Simplify the equation
    expanded_eq = sp.expand(equation)
    simplified_eq = sp.simplify(expanded_eq)

    # Break the equation into terms and process coefficients
    terms = simplified_eq.as_ordered_terms()
    cleaned_terms = []
    for term in terms:
        coeff, rest = term.as_coeff_Mul()
        if abs(coeff) > threshold:
            # Round the coefficient
            rounded_coeff = round(coeff, decimals)
            cleaned_terms.append(rounded_coeff * rest)

    # Recombine the cleaned terms
    simplified_cleaned_eq = sp.Add(*cleaned_terms)

    return simplified_cleaned_eq


# Example usage
x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5')
equation = 0.46 * (-0.0006*x2 + Abs((0.038*x1 - 0.066) * (31.376*x1 + 46.7)) + 
                   0.008 * Abs(-0.003*x1 + 48.753*x1 + 0.839) + 0.475)**2 - 9.96

threshold = 1e-2
decimals = 2

simplified_eq = simplify_and_round_equation(equation, threshold, decimals)
print("Original Equation:", equation)
print("Simplified and Rounded Equation:", simplified_eq)
