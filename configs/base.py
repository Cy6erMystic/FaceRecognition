__all__=['BaseConfig']

class BaseConfig():
    def __init__(self, args: dict = {}) -> None:
        self._replace(args)
    
    def _replace(self, args: dict):
        for k, v in args.items():
            if v is not None:
                setattr(self, k, v)