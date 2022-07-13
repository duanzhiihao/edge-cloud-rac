
_known_models = dict()

def register_model(func):
    name = func.__name__
    if name in _known_models:
        msg = f'Warning: model function *{name}* is multiply defined.'
        print(f'\u001b[93m' + msg + '\u001b[0m')
    _known_models[name] = func
    return func

def get_model(name):
    func = _known_models[name]
    return func
