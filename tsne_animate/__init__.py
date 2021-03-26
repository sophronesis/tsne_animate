import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.animation import FuncAnimation
import sklearn
from sklearn import manifold
from numpy import linalg

class tsneAnimate():
    def __init__(self,tsne):
        self.tsne = tsne
        self.isfit = False
    def getSteps(self,X,y):
        #based on https://github.com/oreillymedia/t-SNE-tutorial
        old_grad = sklearn.manifold._t_sne._gradient_descent
        positions = []
        def _gradient_descent(objective, p0, it, n_iter, objective_error=None,
                              n_iter_check=1, n_iter_without_progress=50,
                              momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                              min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                              args=None, kwargs=None):
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}

            p = p0.copy().ravel()
            update = np.zeros_like(p)
            gains = np.ones_like(p)
            error = np.finfo(np.float).max
            best_error = np.finfo(np.float).max
            best_iter = 0

            for i in range(it, n_iter):
                # We save the current position.
                positions.append(p.copy())

                new_error, grad = objective(p, *args, **kwargs)
                grad_norm = linalg.norm(grad)

                inc = update * grad >= 0.0
                dec = np.invert(inc)
                gains[inc] += 0.05
                gains[dec] *= 0.95
                np.clip(gains, min_gain, np.inf)
                grad *= gains
                update = momentum * update - learning_rate * grad
                p += update

                if (i + 1) % n_iter_check == 0:
                    if new_error is None:
                        new_error = objective_error(p, *args)
                    error_diff = np.abs(new_error - error)
                    error = new_error

                    if verbose >= 2:
                        m = "[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f"
                        print(m % (i + 1, error, grad_norm))

                    if error < best_error:
                        best_error = error
                        best_iter = i
                    elif i - best_iter > n_iter_without_progress:
                        if verbose >= 2:
                            print("[t-SNE] Iteration %d: did not make any progress "
                                  "during the last %d episodes. Finished."
                                  % (i + 1, n_iter_without_progress))
                        break
                    if grad_norm <= min_grad_norm:
                        if verbose >= 2:
                            print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                                  % (i + 1, grad_norm))
                        break
                    if error_diff <= min_error_diff:
                        if verbose >= 2:
                            m = "[t-SNE] Iteration %d: error difference %f. Finished."
                            print(m % (i + 1, error_diff))
                        break

                if new_error is not None:
                    error = new_error

            return p, error, i

        #Replace old gradient func
        sklearn.manifold._t_sne._gradient_descent = _gradient_descent
        X_proj = self.tsne.fit_transform(X)
        self.isfit = True
        #return old gradient descent back
        sklearn.manifold._t_sne._gradient_descent = old_grad
        return positions
    
    def animate(self,X,y,useTqdm=0,filename=None,return_anim=True):
        pos = self.getSteps(X,y)
        y_mapping = {i:n for n,i in enumerate(set(y))}
        
        last_iter = pos[len(pos)-1].reshape(-1, 2)
        lims = np.max(last_iter,axis=0),np.min(last_iter,axis=0)
        NCOLORS = len(y_mapping)
        fig = plt.figure()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        jet = plt.get_cmap('jet') 
        cNorm  = colors.Normalize(vmin=0, vmax=NCOLORS)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        
        A,B = np.array(list(zip(*pos[0].reshape(-1, 2))))
        dots_list = []
        for i in range(NCOLORS):
            colorVal = scalarMap.to_rgba(i)
            a,b = A[y == i],B[y == i]
            dots, = ax.plot(b,a,'o',color=colorVal)
            dots_list.append(dots)
        
        
        def init():
            ax.set_xlim([lims[0][0],lims[1][0]])
            ax.set_ylim([lims[0][1],lims[1][1]])
            return [i for i in dots_list]

        def update(i):
            for j in range(len(dots_list)):
                a,b = np.array(list(zip(*pos[i].reshape(-1, 2))))
                a,b = a[y == j],b[y == j]
                dots_list[j].set_xdata(a)
                dots_list[j].set_ydata(b) 
            return [i for i in dots_list]+[ax]

        if useTqdm==0:
            frames = np.arange(0, len(pos)-1)
        elif useTqdm==1:
            from tqdm import tqdm
            frames = tqdm(np.arange(0, len(pos)-1))
        elif useTqdm==2:
            from tqdm import tqdm_notebook
            frames = tqdm_notebook(np.arange(0, len(pos)-1))
        
        anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=50)
        if return_anim:
            return anim
        if filename==None:
            plt.show()
        else:
            #anim.save(filename, fps=20, codec='libx264')
            anim.save(filename, dpi=80, writer='imagemagick')
