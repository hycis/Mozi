

def valid(x, y, kernel, stride):
    return ((x-kernel)/stride + 1, (y-kernel)/stride + 1)


def full(x, y, kernel, stride):
    return ((x+kernel)/stride - 1, (y+kernel)/stride - 1)


def spp_outdim(input_channels, levels):    
    outdim = 0
    for lvl in levels:
        outdim += lvl**2 * input_channels
    return outdim
