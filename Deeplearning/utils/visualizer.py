import visdom
import numpy as np

class Visualizer():
    def __init__(self, evn):
        self.vis = visdom.Visdom(env=evn)
        self.index = {}

    def plot_stack(self, d, win):
        '''
        :param d: 储存需要显示的数据字典
        :return:
        '''
        name = list(d.keys())

        x = self.index.get(win, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))

        self.vis.line(Y=y, X=np.ones(y.shape) * x,
                      win=win,
                      opts=dict(legend=name,
                                title=win),
                      update=None if x == 0 else 'append'
                      )
        self.index[win] = x + 1