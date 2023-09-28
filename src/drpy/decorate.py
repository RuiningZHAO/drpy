import warnings


def ignoreWarning(func, action, category, lineno=0, append=False):
    
    def wrapper(*args, **kwargs):
        
        with warnings.catch_warnings():
            
            # Filter warnings
            warnings.simplefilter(action, category, lineno, append)
            
            return func(*args, **kwargs)
    
    return wrapper