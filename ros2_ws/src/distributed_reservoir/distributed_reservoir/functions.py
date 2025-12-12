
class dynamical_functions:
    
    @staticmethod
    def return_function(func: str):
        #return reference to the function string input
        functions = {
            'chua': dynamical_functions.chua,
            'rossler': dynamical_functions.rossler,
            'lorenz': dynamical_functions.lorenz
        }
        try:
            return functions[func]
        except:
            print(f"{func} not defined")
            # want to make this fail the ros node gracefully
    
    @staticmethod
    def function_dims(func: str):
        # return output dimension of function
        functions = {
            'chua': 3,
            'rossler': 3,
            'lorenz': 3
        }
        try:
            return functions[func]
        except:
            print(f"{func} not defined")
            # want to make this fail the ros node gracefully
    
    @staticmethod
    def rossler(t, u, a = 0.2, b = 0.2, c = 5.7):
        x, y, z = u
        return [-y - z, 
                x + a * y, 
                b + z * (x - c)]
    
    @staticmethod
    def lorenz(t, u, rho = 28.0, sigma = 10.0, beta = 8/3):
        x, y, z = u
        return [sigma*(y-x),
                x*(rho - z) - y,
                x*y - beta*z]

    @staticmethod
    def chua(t, u, alpha = 15.6, beta = 28.0, m0 = -1.143, m1 = -0.714):
        x, y, z = u
        f_x = m1*x + 0.5*(m0 - m1)*(abs(x + 1) - abs(x - 1))
        return [alpha * (y - x - f_x),
                x - y + z,
                -beta * y]

    
        
    
    