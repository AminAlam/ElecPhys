from functools import wraps
import os

class ErrorHandler:
    def __init__(self):
        pass

    def error(self, e):
        print(e)
        return None

    def error_handler(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get('DEBUG') == 'True':
                return func(*args, **kwargs)
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return self.error(e)
        return wrapper