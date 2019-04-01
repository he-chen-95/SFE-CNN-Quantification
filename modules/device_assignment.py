# by default, all the variable will be placed on '/gpu:0'
# so we need a custom device function, to assign all variables to 'cpu:0'
# note: if GPUs are peered, '/gpu:0' can be a faster option

PS_OPS = ['Variable','VariableV2','AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device
    return _assign