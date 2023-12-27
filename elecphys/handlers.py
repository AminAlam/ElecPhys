from functools import wraps
import os


class ErrorHandler:
    """ Error handler class

        Parameters
        ----------

        Returns
        ----------
    """

    def __init__(self):
        pass

    def error(self, e: str):
        """ Prints error message

        Parameters
            ----------
            e: str
                error message

        Returns
            ----------
            None
        """
        print(e)
        return None

    def error_handler(self, func):
        """ Error handler decorator

        Parameters
            ----------
            func: function
                function to be decorated

        Returns
            ----------
            wrapper: function
                decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get('ELECPHYS_DEBUG') == 'True':
                return func(*args, **kwargs)
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return self.error(e)
        return wrapper
