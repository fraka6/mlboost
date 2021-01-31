import inspect

def get_obj_attrs(obj, exclude=()):
    ''' get object attributs '''
    return [name for name in vars(obj) if not name.startswith('_') and name not in exclude]

def get_class_attrs(cls, exclude=()):
    '''get class attributs '''
    return [name for name, val in inspect.getmembers(cls()) if not name.startswith('_') and 
            name not in exclude and not inspect.ismethod(val) and not inspect.isfunction(val)]

