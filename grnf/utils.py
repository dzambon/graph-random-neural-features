
def bell(i):
    # List of Bell's number
    # _bell = [None, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597, 27644437, 190899322, 1382958545, 10480142147, 82864869804, 682076806159, 5832742205057, 51724158235372, 474869816156751, 4506715738447323, 44152005855084346, 445958869294805289, 4638590332229999353, 49631246523618756274]
    _bell = [None, 1, 2, 5, 15]
    return _bell[i]

def param_shapes(k_in, k_out, feat_in, feat_out, feat_rand):
    """ Generate the shapes that kernel and bias should have. """
    Gamma_lin = bell(k_in + k_out)
    Gamma_bias = bell(k_out)
    
    kernel_shape = (feat_rand, Gamma_lin, feat_out, feat_in)
    if Gamma_bias is None:
        # bias_shape = (feat_rand, feat_out)
        bias_shape = (1, feat_rand, feat_out)
    else:
        # bias_shape = (feat_rand, Gamma_bias, feat_out)
        bias_shape = (1, feat_rand, Gamma_bias, feat_out)
    
    return kernel_shape, bias_shape