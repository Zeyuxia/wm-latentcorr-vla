def get_model(*args, **kwargs):
    from .deploy_policy import get_model as _get_model

    return _get_model(*args, **kwargs)


def eval(*args, **kwargs):
    from .deploy_policy import eval as _eval

    return _eval(*args, **kwargs)


def reset_model(*args, **kwargs):
    from .deploy_policy import reset_model as _reset_model

    return _reset_model(*args, **kwargs)

