import numpy as np

# 降序排列的分类器和数值列表
data = [
    ["randomForest", 1],
    ["randomForest", 0.999710145],
    ["randomForest", 0.999320718],
    ["randomForest", 0.999101085],
    ["mlp", 0.99824104],
    ["randomForest", 0.996982127],
    ["mlp", 0.996365553],
    ["randomForest", 0.995778543],
    ["tree", 0.994444444],
    ["mlp", 0.98894896],
    ["mlp", 0.986386469],
    ["tree", 0.984273618],
    ["tree", 0.980555556],
    ["randomForest", 0.979288246],
    ["tree", 0.977777778],
    ["knn", 0.973227011],
    ["mlp", 0.960475597],
    ["mlp", 0.956898253],
    ["tree", 0.94733061],
    ["tree", 0.94294686],
    ["knn", 0.932514732],
    ["tree", 0.931631702],
    ["knn", 0.923199917],
    ["svm", 0.904484981],
    ["knn", 0.904386963],
    ["knn", 0.903604882],
    ["knn", 0.899538701],
    ["svm", 0.878969579],
    ["mlp", 0.869793504],
    ["multinomialNB", 0.863758412],
    ["svm", 0.837519438],
    ["knn", 0.832669919],
    ["multinomialNB", 0.814014303],
    ["svm", 0.78807472],
    ["multinomialNB", 0.775981008],
    ["multinomialNB", 0.759280109],
    ["welm-lin", 0.755130072],
    ["welm-lin", 0.722172547],
    ["multinomialNB", 0.721460341],
    ["welm-lin", 0.714717599],
    ["multinomialNB", 0.706863332],
    ["welm-lin", 0.706165313],
    ["welm-lin", 0.705638415],
    ["svm", 0.705433033],
    ["elm-lin", 0.699880123],
    ["elm-lin", 0.698096444],
    ["welm-lin", 0.691161713],
    ["elm-lin", 0.688320164],
    ["elm-lin", 0.672884467],
    ["welm-lin", 0.672108034],
    ["elm-lin", 0.665646028],
    ["svm", 0.654876801],
    ["elm-lin", 0.648141948],
    ["svm", 0.637853775],
    ["multinomialNB", 0.618383567],
    ["elm-lin", 0.569970988]
]

# 按照第二列数值降序排序
sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

# 计算每个分类器的平均排名
classifiers = {}
ranks = {}
for i, (classifier, value) in enumerate(sorted_data):
    if classifier not in classifiers:
        classifiers[classifier] = [i + 1]
    else:
        classifiers[classifier].append(i + 1)

for classifier, rank_list in classifiers.items():
    ranks[classifier] = np.mean(rank_list)

print("分类器的平均排名:")
for classifier, rank in ranks.items():
    print(classifier + ": " + str(rank))