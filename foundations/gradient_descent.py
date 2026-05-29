class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # w_new = w_old -  lr * dw
        x = init
        for i in range(iterations):
            dx = 2 * x
            x = x - learning_rate * dx

        return round(x, 5)
            
    