from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(tp, tn, fp, fn):
    confusion_matrix = np.array([[tp, fn], [fp, tn]])
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues',
                xticklabels=['Female', 'Male'],
                yticklabels=['Female', 'Male'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

# ファインチューニングしたモデルをロード
model = YOLO('./best.pt')

# テストデータが含まれるディレクトリを指定（サブディレクトリを含む）
source = '/datasets/test/*/*'

# ディレクトリ内の画像に対して推論を実行
results = model(source, stream=True)  # generator of Results objects

# カウンターの初期化
tp = tn = fp = fn = 0
total = 0

# 結果を処理
for result in results:
    # ファイル名から正解ラベルを取得
    filename = os.path.basename(result.path)
    true_label = 'female' if 'female' in filename else 'male'

    # 予測されたクラスを取得
    predicted_class_index = result.probs.top1
    predicted_class = 'female' if predicted_class_index == 0 else 'male'

    # TP, TN, FP, FN をカウント
    if predicted_class == 'female' and true_label == 'female':
        tp += 1
    elif predicted_class == 'male' and true_label == 'male':
        tn += 1
    elif predicted_class == 'female' and true_label == 'male':
        fp += 1
    elif predicted_class == 'male' and true_label == 'female':
        fn += 1

    total += 1

# 精度と再現率を計算
accuracy = (tp + tn) / total
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


# 混同行列をプロット
plot_confusion_matrix(tp, tn, fp, fn)
