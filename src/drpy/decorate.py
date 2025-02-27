import warnings, asyncio, functools

def filterWarning(action, category, lineno=0, append=False):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            with warnings.catch_warnings():

                # Filter warnings
                warnings.simplefilter(action, category, lineno, append)

                return func(*args, **kwargs)

        return wrapper

    return decorator


def makeAsync(func):

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):

        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(None, func, *args, **kwargs)

    return async_wrapper