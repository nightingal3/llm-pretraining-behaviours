from matplotlib import pyplot as plt
import matplotlib
import numpy as np

font = {'size': 8}

matplotlib.rc('font', **font)

data = [[1.41,1.90,1.68,1.70,1.82,5.66,1.91,2.45,1.80,2.32,2.12,29.84],
        [1.15,1.86,1.18,1.18,1.67,4.31,1.89,2.59,1.21,2.93,2.00,11.46],
        [4.45,5.27,4.52,4.52,4.87,2.89,4.97,5.34,4.55,5.29,4.94,17.45],
        [1.77,1.86,1.63,1.69,1.73,2.22,1.71,2.10,1.65,2.15,1.80,13.57],
        [1.53,1.59,1.40,1.46,1.49,1.92,1.47,1.76,1.42,1.79,1.56,12.08],
        [1.44,1.47,1.31,1.37,1.39,1.79,1.37,1.62,1.33,1.65,1.46,11.40],
        [1.39,1.41,1.27,1.32,1.34,1.72,1.32,1.54,1.28,1.57,1.41,10.98],
        [1.35,1.37,1.23,1.29,1.30,1.66,1.29,1.49,1.25,1.51,1.37,10.66],
        [1.31,1.33,1.21,1.26,1.27,1.62,1.27,1.44,1.22,1.47,1.34,10.39]]

data_with_en = [[1.41,1.90,1.68,1.70,1.82,5.66,1.91,2.45,1.80,2.32,2.12,2.53],
                [1.15,1.86,1.18,1.18,1.67,4.31,1.89,2.59,1.21,2.93,2.00,0.97],
                [1.57,1.88,1.63,1.70,1.74,2.24,1.72,2.13,1.66,2.18,1.80,1.23],
                [1.36,1.60,1.41,1.47,1.50,1.93,1.47,1.78,1.43,1.81,1.57,1.10],
                [1.28,1.48,1.32,1.38,1.39,1.80,1.37,1.64,1.34,1.66,1.47,1.04],
                [1.23,1.42,1.27,1.33,1.34,1.73,1.32,1.56,1.29,1.58,1.41,1.01],
                [1.20,1.37,1.24,1.30,1.30,1.67,1.29,1.50,1.25,1.52,1.37,0.98]]


X = np.arange(12)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'darkorange', width = 0.09)
ax.bar(X + 0.1, data[1], color = 'salmon', width = 0.09)
ax.bar(X + 0.2, data[2], color = 'lightblue', width = 0.09)
ax.bar(X + 0.3, data[3], color = 'lightskyblue', width = 0.09)
ax.bar(X + 0.4, data[4], color = 'deepskyblue', width = 0.09)
ax.bar(X + 0.5, data[5], color = 'cornflowerblue', width = 0.09)
ax.bar(X + 0.6, data[6], color = 'steelblue', width = 0.09)
ax.bar(X + 0.7, data[7], color = 'royalblue', width = 0.09)
ax.bar(X + 0.8, data[8], color = 'navy', width = 0.09)

X = [i+0.4 for i in X]
ax.set_xticks(X, ('EN','DE','ES','FR','IT','KO','NL','PL','PT','RU','SV','ZH'))
ax.set_ylim(1, 2.7)
ax.set_yticks(np.arange(1, 2.7, 0.1))
ax.legend(labels=['LLaMa','BLOOM','32k','64k','96k','128k','160k','192k','224k'])
ax.set_ylabel('Fertility (pieces / word)')

plt.savefig('./fertility_de_es_fr_it_ko_nl_pl_pt_ru_sv_zh.png', transparent=False, bbox_inches='tight', dpi=1000)
plt.close()

X = np.arange(12)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data_with_en[0], color = 'darkorange', width = 0.11)
ax.bar(X + 0.12, data_with_en[1], color = 'salmon', width = 0.11)
ax.bar(X + 0.24, data_with_en[2], color = 'lightblue', width = 0.11)
ax.bar(X + 0.36, data_with_en[3], color = 'lightskyblue', width = 0.11)
ax.bar(X + 0.48, data_with_en[4], color = 'deepskyblue', width = 0.11)
ax.bar(X + 0.60, data_with_en[5], color = 'steelblue', width = 0.11)
ax.bar(X + 0.72, data_with_en[6], color = 'navy', width = 0.11)

X = [i+0.36 for i in X]
ax.set_xticks(X, ('EN','DE','ES','FR','IT','KO','NL','PL','PT','RU','SV','ZH'))
ax.set_ylim(0.9, 2.7)
ax.set_yticks(np.arange(1, 2.7, 0.1))
ax.legend(labels=['LLaMa','BLOOM','64k','96k','128k','160k','192k'])
ax.set_ylabel('Fertility (pieces / word)')

plt.savefig('./fertility_en_de_es_fr_it_ko_nl_pl_pt_ru_sv_zh.png', transparent=False, bbox_inches='tight', dpi=1000)