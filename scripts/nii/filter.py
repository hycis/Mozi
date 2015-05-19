
import numpy as np
model = 'AE0712_Warp_500_20140713_0358_18325883'
spec = '1119_1.spec.unwarp.f8'
format = 'f8'
path = '/Volumes/Storage/generated_specs/Laura/%s/%s'%(model, spec)

data = np.fromfile(path, dtype=format)

new = []
for e in data:
    if e > 1e2:
        new.append(0)
    else:
        new.append(e)

new = np.asarray(new)


new_path = '/Volumes/Storage/generated_specs/Laura/%s/new_%s'%(model, spec)
new.tofile(new_path, format=format)