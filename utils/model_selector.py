import collections
import numpy as np


class ModelSelector:

    def __init__(self):
        self.history = collections.defaultdict(list)

    def log(self, the_dict):
        for key, value in the_dict.items():
            self.history[key].append(value)
        best_info = {}
        for key in self.history:
            # valid/ms_fv
            # => best-valid/valid_ms_fv
            best_info["best-" + key] = max(self.history[key])
        return best_info

    def is_best_model(self, indicator, big_is_better=True):
        assert indicator in self.history
        arr = self.history[indicator]
        if big_is_better:
            return np.argmax(arr) == (len(arr) - 1)
        else:
            return np.argmin(arr) == (len(arr) - 1)

    def should_stop(self, indicator, early_stop=10, big_is_better=True):
        arr = self.history[indicator]
        if big_is_better:
            index = np.argmax(arr)
        else:
            index = np.argmin(arr)

        if len(arr) - 1 - index >= early_stop:
            return True
        return False

    def get_best_step_info(self, indictor, print_it=True):
        index = np.argmax(self.history[indictor])
        ans = {}
        if print_it:
            for key in self.history:
                v = self.history[key][index]
                print("%s\t%.4f" % (key, v))
                ans[key] = v
        return ans
