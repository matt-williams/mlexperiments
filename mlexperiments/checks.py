import os
import numpy as np
import IPython

def check_pil(Image):
    caught_attribute_error = False
    try:
        Image.fromarray(np.zeros((1, 1))).save('tmp.gif', format='gif')
        Image.open('tmp.gif')
        os.remove('tmp.gif')
    except AttributeError:
        app = IPython.Application.instance()
        app.kernel.do_shutdown(True)
        # in case that didn't work...
        raise Exception('PIL initialization failed - restart runtime and rerun from beginning')