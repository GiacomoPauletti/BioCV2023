import numpy as np

class BorderDetector:
    classifier = None
    def __init__(self, border_classifier):
        assert(type(border_classifier) == dict)
        # asserting every combination of pixels is evaluated
        for value in border_classifier.values():
            assert(len(value)==len(border_classifier.keys()))

        self.classifier = border_classifier

    def detect(self, msk):
        border_msk = np.zeros(shape=msk.shape)
        for row in range(1,msk.shape[0]-1):
            for col in range(1,msk.shape[1]-1):
                spotlight = msk[[[row-1],[row],[row+1]], [col-1,col,col+1]].flatten()
                cur_pixel = spotlight[4] 
                spotlight = np.setdiff1d(spotlight, np.array([cur_pixel]))  # removing all the occurences of cur_pixel in the spotlight
                try:
                    
                    if ( spotlight.size != 0 ):
                        border_msk[row][col] = self.classifier[cur_pixel][np.max(spotlight)] 
                    else:
                        border_msk[row][col] = 0
                except Exception as e:
                    raise ValueError(f"Invalid value {np.max(spotlight)}: only possible values for classifier are {list(self.classifier.keys())}").with_traceback(None)
        return border_msk

    # @ requires detect_borders() applied on msk
    def amplify(self, border_msk, num_iter=1, only_if_0=False):
        assert(num_iter > 0)
        big_border_msk = np.array(border_msk, copy=True)
        for n in range(num_iter):
            old_big = np.array(big_border_msk, copy=True)
            for row in range(1,border_msk.shape[0]-1):
                for col in range(1,border_msk.shape[1]-1):
                    spotlight = old_big[[[row-1],[row],[row+1]], [col-1,col,col+1]].flatten()
                    if (not(only_if_0 and big_border_msk[row][col]!=0)):
                        big_border_msk[row][col] = np.max(spotlight)

        return big_border_msk
    
    def detect_and_amplify(self, msk, num_iter=1, only_if_0=False):
        return self.amplify(self.detect(msk), num_iter=num_iter, only_if_0=only_if_0)

    def enhance(self, border_msk, value, outside=True):
        ...
