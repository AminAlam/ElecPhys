from functools import wraps

class ErrorHandler:
    def __init__(self):
        pass

    def error(self, e):
        print(e)
        return None

    def error_handler(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return self.error(e)
        return wrapper