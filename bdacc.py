import numpy as np
import matplotlib.pyplot as plt


def _poly_of_log_integral(poly, xmin, xmax, N=1000):
    degree = poly.shape[0] - 1
    x = np.linspace(xmin, xmax, num=N)
    y = sum([poly[i] * np.log(x)**(degree-i) for i in range(degree + 1)])
    assert y.shape == (N,)
    # approximated area under the curve (ie, definite integral)
    auc = np.sum(y) * float(1/N)
    return auc, (x, y)


def bd_accuracy(stats1, stats2, visualize=False):
    bpp1 = stats1['bpp']
    acc1 = stats1['acc']
    name1 = stats1.get('name', 'input 1')
    bpp2 = stats2['bpp']
    acc2 = stats2['acc']
    name2 = stats2.get('name', 'input 2')

    logb1 = np.log(bpp1)
    logb2 = np.log(bpp2)

    degree = 4
    poly1 = np.polyfit(logb1, acc1, deg=degree)
    poly2 = np.polyfit(logb2, acc2, deg=degree)

    bppmin = max(min(bpp1), min(bpp2))
    bppmax = min(max(bpp1), max(bpp2))
    auc1, (_x1,_y1) = _poly_of_log_integral(poly1, xmin=bppmin, xmax=bppmax, N=10000)
    auc2, (_x2,_y2) = _poly_of_log_integral(poly2, xmin=bppmin, xmax=bppmax, N=10000)

    bd_acc = (auc2 - auc1) / (bppmax - bppmin)

    if visualize:
        x = np.linspace(np.log(bppmin), np.log(bppmax), num=200)
        fig1, ax = plt.subplots(1, 2, figsize=(12,5))
        plt.setp(ax, ylim=(min(min(acc1), min(acc2))-1, max(max(acc1), max(acc2))+1))
        # left figure: log space
        l1 = ax[0].plot(logb1, acc1, label=f'data - {name1}', marker='.', markersize=12, linestyle='none')
        ax[0].plot(x, np.polyval(poly1, x), label=f'polyfit - {name1}', color=l1[0].get_color())
        l2 = ax[0].plot(logb2, acc2, label=f'data - {name2}', marker='.', markersize=12, linestyle='none')
        ax[0].plot(x, np.polyval(poly2, x), label=f'polyfit - {name2}', color=l2[0].get_color())
        ax[0].set_xlabel('$\log R$')
        ax[0].set_ylabel('$A$')
        ax[0].set_title('Polynomial fitting in the log rate space', fontdict = {'fontsize' : 14})
        ax[0].legend(loc='lower right')
        # right figure: normal space
        l1 = ax[1].plot(bpp1,  acc1, label=f'data - {name1}', marker='.', markersize=12, linestyle='none')
        ax[1].plot(_x1, _y1, label=f'polyfit - {name1}', color=l1[0].get_color())
        l2 = ax[1].plot(bpp2,  acc2, label=f'data - {name2}', marker='.', markersize=12, linestyle='none')
        ax[1].plot(_x2, _y2, label=f'polyfit - {name2}', color=l2[0].get_color())
        ax[1].set_xlabel('$R$')
        ax[1].set_ylabel('$A$')
        ax[1].set_title('Mapping back to the normal space', fontdict = {'fontsize' : 14})
        ax[1].legend(loc='lower right')
    else:
        pass

    return bd_acc
