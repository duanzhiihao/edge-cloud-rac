from matplotlib.ticker import ScalarFormatter
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
    # plt.figure(figsize=(7,6))
    fig1, ax1 = plt.subplots(figsize=(7,6))

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

    ax1.plot([0.0, 3.0], [76.15, 76.15], label='ResNet-50, Original',
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

    results = {
        'quality': [1, 8, 16, 32, 64, 80, 96],
        'bpp': [0.233027, 0.361705, 0.466873, 0.661756, 1.028537, 1.418366, 3.293082],
        'acc': [46.67, 56.37, 61.34, 67.39, 70.546, 72.138, 73.956],
        'acc-ft': [68.06, 71.01, 72.41, 73.95, 75.25, 75.74, None],
    }
    # plot(results['bpp'], results['acc'], label='WebP, no fine-tune')
    plot(results['bpp'], results['acc-ft'], label='WebP, fine-tune')

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

    wacv2022_compression = {
        'top1': [57.37, 71.608, 74.226, 75.096, 75.314, 75.632, 75.928, 75.916],
        'top5': [80.74, 90.554, 91.902, 92.4, 92.584, 92.68, 92.862, 92.864],
        'bpp': [0.148989, 0.333202, 0.598442, 0.889434, 1.314741, 1.834876, 2.595726, 3.657453],
    }
    # plot(wacv2022_compression['bpp'], wacv2022_compression['top1'], label='Baseline (practice)')

    wacv2022 = {
        'acc': [71.42, 73.36, 74.66],
        'bpp': [0.3237, 0.5818, 0.8305]
    }
    # plot(wacv2022['bpp'], wacv2022['acc'], label='Baseline (my impl., last week)')
    wacv2022 = {
        'acc': [67.95, 70.9, 72.6, 74.03, 74.75, 76.21],
        'bpp': [0.2204, 0.2761, 0.4072, 0.5741, 0.8367, 3.267]
    }
    plot(wacv2022['bpp'], wacv2022['acc'], label='Baseline (my impl.)')

    wacv2022 = {
        'acc': [73.06],
        'bpp': [0.7058]
    }
    # plot(wacv2022['bpp'], wacv2022['acc'], label='Baseline (my impl. 2-stage)')

    wacv2022 = {
        'acc': [74.41],
        'bpp': [0.7497]
    }
    # plot(wacv2022['bpp'], wacv2022['acc'], label='Baseline (my impl. cosine decay)')

    # wacv2022 = {
    #     'acc': [76.21],
    #     'bpp': [3.267]
    # }
    # plot(wacv2022['bpp'], wacv2022['acc'], label='Baseline (my impl. v2)')

    vq = {
        'acc': [71.07, 71.96, 72.85],
        'bpp': [0.2883, 0.3467, 0.4636]
    }
    # plot(vq['bpp'], vq['acc'], label='Baseline w/ VQ x4')

    vq = {
        'acc': [69.59, 69.18],
        'bpp': [0.1366, 0.168]
    }
    # plot(vq['bpp'], vq['acc'], label='Baseline w/ VQ x8')

    rate_acc = {
        'bpp': [0.09236, 0.1098, 0.1489, 0.1745, 0.2124, 0.3188, 0.442, 0.7213, 1.194],
        'acc': [67.27, 69.27, 68.52, 69.99, 71.53, 72.93, 73.64, 74.88, 75.1],
    }
    plot(rate_acc['bpp'][2:], rate_acc['acc'][2:], label='Ours s8')
    # plot(rate_acc['bpp'][2:], rate_acc['acc'][2:], label='Baseline w/ encoder 8x')

    rate_acc = {
        'bpp': [0.09303, 0.1125, 0.1443, 0.1754, 0.2143, 0.3236, 0.4604, 0.7328, 1.222, 2.293],
        'acc': [67.03, 68.19, 69.25, 69.85, 71.14, 72.63, 73.55, 74.72, 75.17, 75.27],
    }
    plot(rate_acc['bpp'][2:], rate_acc['acc'][2:], label='Ours s8 small')
    # plot(rate_acc['bpp'][2:], rate_acc['acc'][2:], label='Baseline w/ encoder 8x small')

    rate_acc = {
        'bpp': [0.1786, 0.2829, 0.4206, 0.5508, 0.7209, 1.178],
        'acc': [67.83, 70.69, 73.06, 73.84, 74.07, 74.64],
    }
    plot(rate_acc['bpp'], rate_acc['acc'], label='Ours s8 tiny')
    # plot(rate_acc['bpp'], rate_acc['acc'], label='Baseline w/ encoder 8x tiny')

    rate_acc = {
        'bpp': [0.1532, 0.2359],
        'acc': [68.25, 70.79],
    }
    # plot(rate_acc['bpp'], rate_acc['acc'], label='Ours s8 next')

    rate_acc = {
        'bpp': [0.1138, 0.1492, 0.1797, 0.242, 0.3843, 0.6799],
        'acc': [65.01, 66.13, 67.99, 69.7, 71.98, 73.23],
    }
    # plot(rate_acc['bpp'][1:], rate_acc['acc'][1:], label='Baseline w/ encoder 16x')

    ours = {
        'acc': [73.28, 75.22],
        'bpp': [0.5347, 0.97605]
    }
    # plot(ours['bpp'], ours['acc'], label='Baseline w/ mobilenet encoder')

    ours = {
        'acc': [72.74, 75.6],
        'bpp': [0.6789, 1.456]
    }
    # plot(ours['bpp'], ours['acc'], label='Baseline w/ convnext encoder')

    plt.title('Rate-accuracy trade-off on ImageNet')
    plt.grid(True, alpha=0.32)
    plt.legend(loc='best')
    plt.xlabel('Bits per pixel (bpp)', fontdict=default_font)
    plt.xscale('log')
    plt.xlim(0.13, 2.0)
    plt.xticks([0.13, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    # plt.yticks(list(range()))
    # plt.ylim(22, 42)
    plt.ylabel('Top-1 acc.', fontdict=default_font)
    # plt.title('Rate-distortion curves')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_mobilecloud()
