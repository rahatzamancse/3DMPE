"""Gradient-descent step rules and drivers used by MDS/MPSE.

Each step rule takes the current position/gradient pair ``(x, dfx)`` plus
scheme-specific state and returns the next position along with a dictionary of
bookkeeping values (learning rate, step size, a ``stop`` flag, etc.).
"""

import math
import time
import matplotlib.pyplot as plt
import numpy as np


def fixed(x, dfx, lr=1.0, p=None, **kwargs):
    """Fixed learning-rate gradient step (with optional projection ``p``)."""
    dx = -lr * dfx
    ndx = np.linalg.norm(dx)
    y = x + dx
    if p is None:
        x = y
        step = ndx
    else:
        x_new = p(y)
        step = np.linalg.norm(x_new - x)
        x = x_new
    out = {'lr': lr, 'ndx': ndx, 'y': y, 'step': step, 'df0x0': dfx,
           'stop': False}
    return x, out


def bb(x, dfx, x0=0, dfx0=0, p=None, y=None, **kwargs):
    """Barzilai-Borwein (1988) adaptive gradient step."""
    x_initial = x
    ddfx = dfx - dfx0
    nddfx = np.linalg.norm(ddfx)
    if nddfx == 0.0:
        out = {'stop': True}
    else:
        if y is None:
            y = x
        diff = y - x0
        ndx = np.linalg.norm(diff)
        lr = abs(np.sum(diff * ddfx)) / nddfx ** 2
        x0 = x
        dfx0 = dfx
        dx = -lr * dfx
        y = x + dx
        if p is None:
            step = ndx
            x = y
        else:
            x = p(y)
            step = np.linalg.norm(x - x_initial)
        out = {'lr': lr, 'ndx': ndx, 'step': step, 'y': y, 'x0': x0,
               'dfx0': dfx0, 'stop': False}
    return x, out


def mm(x, dfx, df0x=None, x0=0, df0x0=0, p=None, y=0, ndx=None, lr=10,
       theta=np.inf, alpha=1.0, **kwargs):
    """Malitsky-Mishchenko (2019) adaptive gradient step (with projection)."""
    if ndx is None:
        if p is None:
            ndx = np.linalg.norm(x - x0)
        else:
            ndx = np.linalg.norm(y - x0)
    if ndx == 0:
        out = {'stop': True}
    else:
        if df0x is None:
            df0x = dfx
        nddfx = np.linalg.norm(df0x - df0x0)

        L = nddfx / ndx
        lr0 = lr
        lr = max(min(math.sqrt(1 + theta) * lr, 1 / (alpha * L)), 3 * lr / 4)
        theta = lr / lr0
        dx = -lr * dfx
        ndx = np.linalg.norm(dx)
        y = x + dx
        if p is None:
            step = ndx
            x = y
        else:
            x0 = x
            x = p(y)
            step = np.linalg.norm(x - x0)
        out = {'ndx': ndx, 'df0x0': dfx, 'lr': lr, 'step': step,
               'theta': theta, 'y': y, 'stop': False}
    return x, out


def adam(x, dfx, ndx=0, dfx0=0, lr=0.1, m=0, v=0, i=0, **kwargs):
    """ADAM gradient step."""
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m = beta1 * m + (1 - beta1) * dfx
    v = beta2 * v + (1 - beta2) * dfx ** 2
    mc = m / (1 - beta1 ** (i + 1))
    vc = v / (1 - beta2 ** (i + 1))
    dx = -lr * mc / (np.sqrt(vc) + epsilon)
    ndx = np.linalg.norm(dx)
    x = x + dx
    out = {'lr': lr, 'ndx': ndx, 'm': m, 'v': v, 'i': i + 1, 'stop': False}
    return x, out


schemes = {
    'fixed': fixed,
    'bb': bb,
    'mm': mm,
    'adam': adam,
}


def single(x, F, Xi=None, p=None, scheme='mm', min_cost=None, min_grad=None,
           min_step=None, max_iter=100, max_step=1e10, lr=1, verbose=0,
           indent='', plot=False, **kwargs):
    """Gradient descent for a single parameter array.

    Parameters
    ----------
    x : ndarray
        Initial point.
    F : callable
        Returns ``(gradient, cost)`` at a point. If ``Xi`` is ``None`` the call
        is ``F(x)``; otherwise it is ``F(x, **xi)`` with stochastic parameters.
    Xi : None or callable
        Produces stochastic parameters ``xi`` for stochastic gradient descent.
    p : None or callable
        If given, updates are projected via ``p`` (constrained optimization).
    scheme : str
        Step rule, one of :data:`schemes`.
    """
    stochastic = Xi is not None
    constraint = p is not None
    assert scheme in schemes
    algorithm = schemes[scheme]

    if verbose > 0:
        print(indent + 'gd.single(): ')
        print(indent + '  computation parameters:')
        print(indent + f'    stochastic : {stochastic}')
        print(indent + f'    constraint : {constraint}')
        print(indent + f'    scheme : {scheme}')
        print(indent + f'    initial lr : {lr:0.2e}')
        if min_cost is not None:
            print(indent + f'    min_cost : {min_cost:0.2e}')
        if min_grad is not None:
            print(indent + f'    min_grad : {min_grad:0.2e}')
        if min_step is not None:
            print(indent + f'    min_step : {min_step:0.2e}')
        print(indent + f'    max_iter : {max_iter}')
        print(indent + f'    max_step : {max_step:0.2e}')

    if min_cost is None:
        min_cost = -np.inf
    if min_grad is None:
        min_grad = -np.inf
    if min_step is None:
        min_step = -np.inf

    costs = np.empty(max_iter)
    grads = np.empty(max_iter)
    steps = np.empty(max_iter)
    lrs = np.empty(max_iter)

    success = True
    conclusion = 'maximum number of iterations reached'

    t0 = time.time()

    normalization = math.sqrt(np.size(x))

    x0 = x.copy()
    it0 = 1
    for i in range(it0):
        if stochastic is False:
            dfx, fx = F(x)
        else:
            xi = Xi()
            dfx, fx = F(x, **xi)
        x, kwargs = fixed(x, dfx, lr=lr, p=p)
        costs[i] = fx
        grads[i] = np.linalg.norm(dfx) / normalization
        lrs[i] = kwargs['lr']
        steps[i] = kwargs['ndx'] / normalization
    if constraint is True:
        y = kwargs['y']

    if verbose > 1:
        print(indent + '  progress:')
        print(indent + '    iter:      cost:     grad:     lr:       step:')

    for i in range(it0, max_iter):

        if stochastic is False:
            if constraint is True:
                if scheme in ['bb', 'mm']:
                    dfy, fy = F(y)
                    kwargs['df0x'] = dfy
            dfx, fx = F(x)
        else:
            if constraint is False:
                if scheme in ['bb', 'mm']:
                    df0x, f0x = F(x, **xi)
                    kwargs['df0x'] = df0x
            else:
                if scheme in ['bb', 'mm']:
                    fy, dfy = F(y, **xi)
                    kwargs['df0x'] = dfy
            xi = Xi()
            dfx, fx = F(x, **xi)
        costs[i] = fx
        grads[i] = np.linalg.norm(dfx) / normalization

        if fx < min_cost:
            conclusion = 'minimum cost reached'
            lrs[i] = None
            steps[i] = None
            break
        if grads[i] < min_grad:
            conclusion = 'minimum gradient size reached'
            lrs[i] = None
            steps[i] = None
            break

        x, kwargs = algorithm(x, dfx, p=p, **kwargs)

        if constraint is True:
            y = kwargs['y']
        if kwargs['stop'] is True:
            conclusion = 'update rule'
            break
        lrs[i] = kwargs['lr']
        lr = lrs[i]
        steps[i] = kwargs['ndx'] / normalization
        if steps[i] < min_step:
            conclusion = 'minimum step reached reached'
            break
        elif steps[i] > max_step:
            success = False
            conclusion = 'maximum step size reached (unstable)'
            break
        if verbose > 1:
            print(indent + f'    {i + 1:>4}/{max_iter:>4}  {costs[i]:0.2e}' +
                  f'  {grads[i]:0.2e}  {lrs[i]:0.2e}' +
                  f'  {steps[i]:0.2e}', flush=True, end="\r")

    tf = time.time()

    costs = costs[0:i + 1]
    grads = grads[0:i + 1]
    lrs = lrs[0:i + 1]
    steps = steps[0:i + 1]

    if plot is True:
        fig, ax = plt.subplots()
        ax.semilogy(costs, label='cost', linewidth=3)
        ax.semilogy(grads, label='gradient size', linestyle='--')
        ax.semilogy(lrs, label='learning rate', linestyle='--')
        ax.semilogy(steps, label='step size', linestyle='--')
        ax.legend()
        ax.set_xlabel('iterations')
        plt.draw()
        plt.pause(0.1)

    outputs = {
        'costs': costs,
        'steps': steps,
        'grads': grads,
        'lrs': lrs,
        'iterations': i + 1,
        'success': success,
        'conclusion': conclusion,
        'time': tf - t0,
        'lr': lr,
    }

    if verbose > 0:
        if verbose > 1:
            print()
        print(indent + '  results:')
        print(indent + f'    conclusion : {conclusion}')
        print(indent + f'    total iterations : {i + 1}')
        print(indent + f'    final cost : {costs[-1]:0.2e}')
        print(indent + f'    final gradient size : {grads[-1]:0.2e}')
        print(indent + f'    final learning rate : {lrs[-1]:0.2e}')
        print(indent + f'    final step size : {steps[-1]:0.2e}')
        print(indent + f'    time : {tf - t0:0.2e} [sec]')

    return x, outputs


def multiple(X, F, Xi=None, p=None, scheme='mm', min_cost=None, min_grad=None,
             min_step=None, max_iter=100, max_step=1e10, lr=1, verbose=0,
             indent='', plot=False, **kwargs):
    """Gradient descent optimizing a list of parameter arrays jointly."""
    assert isinstance(X, list)
    K = len(X)

    stochastic = Xi is not None

    if isinstance(p, list):
        assert len(p) == K
    else:
        p = [p] * K
    projected = [pk is not None for pk in p]
    constraint = True in projected

    if isinstance(scheme, list):
        assert len(scheme) == K
    else:
        scheme = [scheme] * K
    if isinstance(lr, list):
        assert len(lr) == K
    else:
        lr = [lr] * K

    algorithm = [schemes[scheme[k]] for k in range(K)]

    if verbose > 0:
        print(indent + 'gd.multiple(): ')
        print(indent + '  computation parameters:')
        print(indent + f'    stochastic : {stochastic}')
        print(indent + f'    constraint : {constraint}')
        print(indent + f'    projected : {projected}')
        print(indent + f'    scheme : {scheme}')
        lrs = ', '.join(f'{a:0.2e}' for a in lr)
        print(indent + f'    initial lr : {lrs}')
        if min_cost is not None:
            print(indent + f'    min_cost : {min_cost:0.2e}')
        if min_grad is not None:
            print(indent + f'    min_grad : {min_grad:0.2e}')
        if min_step is not None:
            print(indent + f'    min_step : {min_step:0.2e}')
        print(indent + f'    max_iter : {max_iter}')
        print(indent + f'    max_step : {max_step:0.2e}')

    if min_cost is None:
        min_cost = -np.inf
    if min_grad is None:
        min_grad = -np.inf
    if min_step is None:
        min_step = -np.inf

    costs = np.empty(max_iter)
    grads = np.empty((max_iter, K))
    steps = np.empty((max_iter, K))
    lrs = np.empty((max_iter, K))

    success = True
    conclusion = 'maximum number of iterations reached'

    t0 = time.time()

    normalization = [math.sqrt(np.size(a)) for a in X]

    it0 = 1
    for i in range(it0):
        if stochastic is False:
            dfX, fX = F(X)
        else:
            xi = Xi()
            dfX, fX = F(X, **xi)
        KWARGS = []
        if constraint is True:
            Y = []
        for k in range(K):
            X[k], temp = fixed(X[k], dfX[k], p=p[k], lr=lr[k])
            KWARGS.append(temp)
            if constraint is True:
                Y.append(temp['y'])
        costs[i] = fX
        grads[i] = [np.linalg.norm(dfX[k]) / normalization[k] for k in range(K)]
        lrs[i] = lr
        steps[i] = [KWARGS[k]['ndx'] / normalization[k] for k in range(K)]

    if verbose > 0:
        print(indent + '  progress:')
        print(indent + '    iter:      cost:     grad:     lr:       step:')

    for i in range(it0, max_iter):

        if stochastic is False:
            if constraint is True:
                dfY, fY = F(Y)
            dfX, fX = F(X)
        else:
            if constraint is False:
                df0X, f0X = F(X, **xi)
            else:
                df0Y, f0Y = F(Y, **xi)
            xi = Xi()
            dfX, fX = F(X, **xi)

        costs[i] = fX
        grads[i] = [np.linalg.norm(a) / b for a, b in zip(dfX, normalization)]

        if fX < min_cost:
            conclusion = 'minimum cost reached'
            lrs[i] = [None] * K
            steps[i] = [None] * K
            break
        if max(grads[i]) < min_grad:
            conclusion = 'minimum gradient size reached'
            lrs[i] = [None] * K
            steps[i] = [None] * K
            break

        for k in range(K):
            if stochastic is False:
                if projected[k] is False:
                    KWARGS[k]['df0x'] = dfX[k]
                else:
                    KWARGS[k]['df0x'] = dfY[k]
            else:
                if constraint is False:
                    KWARGS[k]['df0x'] = df0X[k]
                else:
                    KWARGS[k]['df0x'] = df0Y[k]

            X[k], KWARGS[k] = algorithm[k](X[k], dfX[k], p=p[k], **KWARGS[k])
            if constraint is True:
                Y = [KWARGS[k]['y'] for k in range(K)]
            if KWARGS[k]['stop'] is True:
                conclusion = 'update rule'
                break
            lrs[i, k] = KWARGS[k]['lr']
            lr[k] = lrs[i, k]
            steps[i, k] = KWARGS[k]['ndx'] / normalization[k]
        if max(steps[i]) < min_step:
            conclusion = 'minimum step reached reached'
            break
        elif max(steps[i]) > max_step:
            success = False
            conclusion = 'maximum step size reached (unstable)'
            break
        if verbose > 0:
            print(indent + f'    {i + 1:>4}/{max_iter:>4}  {costs[i]:0.2e}' +
                  f'  {np.max(grads[i]):0.2e}  {np.max(lrs[i]):0.2e}' +
                  f'  {np.max(steps[i]):0.2e}', flush=True, end="\r")

    tf = time.time()

    costs = costs[0:i + 1]
    grads = grads[0:i + 1]
    lrs = lrs[0:i + 1]
    steps = steps[0:i + 1]

    if plot is True:
        fig, axes = plt.subplots(1, 1 + K, figsize=(15, 5))
        axes[0].semilogy(costs, linewidth=3)
        axes[0].set_title('cost')
        for k in range(K):
            axes[k + 1].semilogy(grads[:, k], label='gradient size', linestyle='--')
            axes[k + 1].semilogy(lrs[:, k], label='learning rate', linestyle='--')
            axes[k + 1].semilogy(steps[:, k], label='step size', linestyle='--')
            axes[k + 1].set_title(f'coordinate {k}')
            axes[k + 1].legend()
            axes[k + 1].set_xlabel('iterations')
        plt.draw()
        plt.pause(0.1)

    outputs = {
        'costs': costs,
        'steps': steps,
        'grads': grads,
        'lrs': lrs,
        'iterations': i + 1,
        'success': success,
        'conclusion': conclusion,
        'time': tf - t0,
        'lr': lr,
    }

    if verbose > 0:
        if verbose > 1:
            print()
        print(indent + '  results:')
        print(indent + f'    conclusion : {conclusion}')
        print(indent + f'    total iterations : {i + 1}')
        print(indent + f'    final cost : {costs[-1]:0.2e}')
        print(indent + f'    time : {tf - t0:0.2e} [sec]')
    return X, outputs
