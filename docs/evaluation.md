- [ホールドアウト検証とは？ 10分でわかりやすく解説](https://www.netattest.com/hold-out-validation-2024_mkt_tst)

### Evaluation, Performance Analysis

1. Data Splitting (データ分割): データセットを訓練用 (training)、検証用 (validation)、およびテスト用 (testing) に分割すること。
2. Performance Metrics (評価指標): モデルの性能を測定するための指標（例: 精度 (accuracy), 再現率 (recall), 適合率 (precision), F1スコア）。
3. Cross-Validation (交差検証): モデルがさまざまなデータセットでどのようにパフォーマンスを発揮するかを評価するための手法。
4. Hyperparameter Tuning (ハイパーパラメータ調整): モデルのパフォーマンスを最適化するためにハイパーパラメータを調整するステップ。
5. Comparison and Benchmarking (比較とベンチマーク): 複数のモデルを比較して、どのモデルがタスクに最も適しているかを判断すること。

### バリデーション (Validation or Evaluation?)

* 訓練用と検証用のデータに分けて機械学習モデルの性能を評価するプロセス。検証用のデータで予測をして目的変数と比較してまともな予測ができているかを確認する。
* データセット全体を訓練用とデータを訓練用と検証用にどう分けるのか。
* 全体データセットの目的変数の分布とできるだけ一致しているべき。検証データの目的変数の分布が、現実世界のデータや運用時の分布を反映していない場合、モデルの評価結果が現実と乖離する可能性がある。

### ホールドアウト法
* 機械学習モデルの性能を評価するための手法の一つ。一般的にはデータセットの全体のうち7~8割を訓練データ、残りの2~3割をテストデータとして使用してモデルを評価する方法。
  * [ホールドアウト検証とは？ 10分でわかりやすく解説](https://www.netattest.com/hold-out-validation-2024_mkt_tst)
* ホールドアウト法の問題はテストデータの分布が訓練データと大きく異なる場合にテストデータの評価が適切に行えない。その場合は交差検証法（クロスバリデーション）を使う。クロスバリデーション法では学習も複数回行うので、検証法というより学習/検証法と言ったほうが合っている気がする。


### 評価指標

- 正解率 (Accuracy)
- 不均衡なデータの場合
  - ROC-AUC
  - Precision, Recall
  - F1-Score
  - Balanced Accuracy
