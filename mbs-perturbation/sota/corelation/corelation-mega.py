import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Feature': ['codeSize', 'entityNum', 'entityAttributeNum', 'aveEntityAttribute', 'controllerNum', 'interfaceNum', 'abstractClassNum', 'serviceClassNum', 'dtoObjectNum', 'APINum', 'APIVersionNum', 'maxParaNum', 'serviceImplCallNum', 'maxServiceCall', 'serviceCallPer', 'maxServiceCalled', 'serviceCalledPer'],
    'MS': [0.6128402184032483, -0.14576245486590575, -0.12768754907378566, -0.2676473789302238, 0.6927792839255417, 0.041941818727662145,
             0.4738705654686398, 0.07916577241272621, -0.005476724028345429, 0.6560170317164604, -0.18952932671438616, 0.43084558379945465, 0.47372787043510095, -0.2568562043471911, -0.26393051731473705, -0.30883284926154925, -0.3075866740024059],
    'NS': [-0.4476041271592156, -0.3159478863689382, -0.32014352895877657, -0.2787845255877925, -0.33638133532474906, -0.3407924460513653, -0.14804330818062708, -0.2513908149980848, -0.04841393319356762, -0.4179641641410614, -0.12269938783093591, -0.2914078731888124, -0.3447361355849478,
             -0.11986946392236535, -0.09100876312303796, -0.11901252309900817, -0.10604421569688144],
    'MG': [-0.14409509785236885, -0.10928189155467478, -0.10983191251049265, -0.09396338090746637, -0.08801750302799585, -0.09262856698998864, -0.05188876130757205, -0.03103889805596141, -0.040145117712488805, -0.13030791329321578, 0.04936811188126516, 0.003512308045023927, -0.07962077219602905, -0.0728316109841826, -0.07257949394018211, 0.11802901742501577, 0.13598149553207603],
    'BS': [0.29115766918015173, 0.1771660692934542, 0.17501521379243234, -0.09059722749081822, 0.36040137164557295, 0.17602988692701205, 0.2501024429761968, -0.12836192052067155, 0.07765285381841955, 0.3511742023854986, 0.11639760083682028, 0.37375709468105084, 0.16800420092241672, -0.13030690993049954, -0.12541552176217338, -0.14490639286992332, -0.14150955152946595],
    'CS': [0.050999712775903414, -0.10865652838425953, -0.10432032129527406, -0.07698364455877882, 0.10288529321396908, 0.005730382433191435, -0.05764064470474816, 0.014419998370296188, 0.191723004154878, 0.1207874781907887, -0.033904281635833505, 0.010823590566147831, 0.041796840229158784, 0.24229256450929357, 0.2228895708311631, -0.05488376256153268, -0.051800095082190876],
    'BOS': [-0.05479288231552002, 0.04166557101296963, 0.05347846531593304, 0.15036503812150898, -0.009852353691610362, 0.06953764905331634, -0.14134223066301244, 0.07815301974314219, 0.028278010509970197, 0.05484842206824041, 0.0003968984330242454, -0.026064193686089875, 0.07068166691079786, -0.07406950985435821, -0.0906892074805278, 0.4739949658130387, 0.5428145753084821],
    'SC': [-0.04825129109149388, 0.017648427160011648, 0.04694247045767869, 0.1400818818635166, -0.005761389057736489, -0.009799622876994691, -0.05764064470474816, -0.03992323342749214, 0.0043727621867805455, 0.09104550533581224, 0.013507357092592587, -0.021556607152662624, 0.1025962875218005, 0.15700431436053394, 0.22063975579001271, 0.22521906226478658, 0.21927477971624929]
}

df = pd.DataFrame(data)

# 将特征作为行索引
df.set_index('Feature', inplace=True)

# 绘制热力图
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, ax=ax)
ax.set_ylim(len(df), 0)  # 翻转y轴坐标，使特征名称从上到下显示
plt.tight_layout(rect=[0, 0, 1, 1])  # 调整子图布局，设置右边空白为0.15

plt.savefig('heatmap.jpg', dpi=300)
plt.show()