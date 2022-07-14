import json
import matplotlib.pyplot as plt
from pathlib import Path


default_font = {
    'size': 12,
}

def plot(stat, ax=None):
    x = stat['bpp']
    y = [f*100 for f in stat['top1']]
    label = stat['name']
    module = ax or plt
    p = module.plot(x, y, label=label,
        marker='.', markersize=10, linewidth=1.6,
    )
    return p


def main():
    fig1, ax = plt.subplots(figsize=(8,8))

    results_dir = Path('results')
    all_methods_results = []
    for fpath in results_dir.rglob('*.json'):
        with open(fpath, 'r') as f:
            results = json.load(f)
        results['name'] = fpath.stem
        all_methods_results.append(results)

    for results in all_methods_results:
        plot(results, ax=ax)

    plt.title('Rate-accuracy trade-off on ImageNet')
    plt.grid(True, alpha=0.32)
    plt.legend(loc='lower right', prop={'size': 16})
    plt.xlabel('Bits per pixel (bpp)', fontdict=default_font)
    plt.ylabel('Top-1 acc. (%)', fontdict=default_font)
    x_ticks = [(i) / 10 for i in range(17)]
    plt.xticks(x_ticks)
    y_ticks = [i for i in range(50, 78, 2)]
    plt.yticks(y_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.ylim(min(y_ticks), max(y_ticks))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
