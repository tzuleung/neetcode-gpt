class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        ans = init 

        for _ in range(iterations):
            # x_new = x_old - alpha * gradient
            ans = ans - learning_rate * (2*ans)
        
        return round(ans, 5)

