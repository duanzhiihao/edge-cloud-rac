import matplotlib.pyplot as plt


default_font = {
    # 'family': 'serif',
    # 'color':  'darkred',
    # 'weight': 'normal',
    'size': 12,
}

def plot(x, y, label, ls='-', ax=None):
    module = ax if ax is not None else plt
    p = module.plot(x, y, label=label,
                 marker='.', markersize=10,
                 linestyle=ls, linewidth=1.6)
    return p


def plot_mobilecloud():
    plt.figure(figsize=(7,6))
    jpeg_mobilev3 = {
        'acc': [0.6623, 0.711, 0.7325, 0.7385, 0.7409],
        'bpp': [0.22366, 0.31079, 0.46838, 0.60333, 0.66786]
    }
    # jpeg_mobilev3['acc'].append(0.7507)
    # jpeg_mobilev3['bpp'].append(6.342112)
    # plot(jpeg_mobilev3['bpp'], jpeg_mobilev3['acc'], label='JPEG + MobileNetV3-L')

    factorized = {
        'acc': [0.717, 0.729],
        'bpp': [0.6396, 0.7498]
    }
    # plot(factorized['bpp'], factorized['acc'], label='MobileNetV3-L + Factorized entropy model')

    plt.plot([0.0, 3.0], [76.15, 76.15], label='ResNet-50, Original',
             marker='', linestyle='--', linewidth=1.6, color='gray')

    jpeg_res50 = {
        'acc': [72.57, 71.27, 67.71, 60.58],
        'bpp': [0.6030, 0.4682, 0.3107, 0.2236]
    }
    plot(jpeg_res50['bpp'], jpeg_res50['acc'], label='JPEG enc -> dec -> ResNet-50')
    jpeg_res50_wacv2022 = {
        'acc': [None, None, 50, 64, 68, 69.6, 71, 72, 72.6, 73.4],
        'kbyte': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }
    jpeg_res50_wacv2022['bpp'] = [kb*1024*8 / (224*224) for kb in jpeg_res50_wacv2022['kbyte']]
    # plot(jpeg_res50_wacv2022['bpp'], jpeg_res50_wacv2022['acc'], label='JPEG dec. + ResNet-50 (Irvine)')

    ours_tcsvt = {
        'acc': [73.79, 72.82, 68.37],
        'bpp': [0.595, 0.321, 0.133]
    }
    # plot(ours_tcsvt['bpp'], ours_tcsvt['acc'], label='Our TCSVT')

    wacv2022_paper = {
        'acc': [57.5, 71, 73, 74.5, 74.8, 75, 75.2, 75.4, 75.6, 75.8],
        'kbyte': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }
    wacv2022_paper['bpp'] = [kb*1024*8 / (224*224) for kb in wacv2022_paper['kbyte']]
    # plot(wacv2022_paper['bpp'], wacv2022_paper['acc'], label='Baseline: Irvine 2022 WACV (paper)')

    wacv2022_code = {
        'acc': [0.5737,    0.71608,  0.74226,  0.75096,  0.75314,  0.75632,  0.75928,  0.75916],
        'bpp': [0.1465677, 0.330722, 0.595901, 0.886827, 1.311912, 1.831621, 2.592045, 3.653037],
        'beta': [1.28,     0.64,     0.32,     0.16,     0.08,     0.04,     0.02,     0.01]
    }
    wacv2022_code['acc'] = [f*100 for f in wacv2022_code['acc']]
    plot(wacv2022_code['bpp'], wacv2022_code['acc'], label='Baseline (Irvine 2022 WACV)')

    wacv2022 = {
        'acc': [71.42, 73.25, 74.66],
        'bpp': [0.3237, 0.5742, 0.8305]
    }
    plot(wacv2022['bpp'], wacv2022['acc'], label='Baseline (my impl.)')

    # wacv2022 = {
    #     'acc': [76.21],
    #     'bpp': [3.267]
    # }
    # plot(wacv2022['bpp'], wacv2022['acc'], label='Baseline (my impl. v2)')

    vq = {
        'acc': [71.07, 71.96, 72.85],
        'bpp': [0.2883, 0.3467, 0.4636]
    }
    plot(vq['bpp'], vq['acc'], label='Baseline w/ VQ')

    ours = {
        'acc': [73.28, 75.22],
        'bpp': [0.5347, 0.97605]
    }
    plot(ours['bpp'], ours['acc'], label='Baseline w/ mobilenet encoder')

    plt.title('Rate-accuracy trade-off on ImageNet')
    plt.grid(True, alpha=0.32)
    plt.legend(loc='best')
    plt.xlabel('Bits per pixel (bpp)', fontdict=default_font)
    # plt.xscale('log')
    plt.xlim(0.1, 1.6)
    # plt.yticks(list(range()))
    # plt.ylim(22, 42)
    plt.ylabel('Top-1 acc.', fontdict=default_font)
    # plt.title('Rate-distortion curves')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_mobilecloud()
