~~~python

PS C:\Users\Server\Documents\dysgraphia-detector\ml\pen_ml_model> python .\train_mk.py
Accuracy: 0.875
Precision: 0.0
Recall: 0.0
F1 Score: 0.0
Confusion Matrix:
 [[7 0]
 [1 0]]
              precision    recall  f1-score   support

      Normal       0.88      1.00      0.93         7
  Dysgraphia       0.00      0.00      0.00         1

    accuracy                           0.88         8
   macro avg       0.44      0.50      0.47         8
weighted avg       0.77      0.88      0.82         8

PS C:\Users\Server\Documents\dysgraphia-detector\ml\pen_ml_model> python .\train_model.py
Accuracy: 0.875
Precision: 0.0
Recall: 0.0
F1: 0.0
Confusion Matrix:
 [[7 0]
 [1 0]]
              precision    recall  f1-score   support

      Normal       0.88      1.00      0.93         7
  Dysgraphia       0.00      0.00      0.00         1

    accuracy                           0.88         8
   macro avg       0.44      0.50      0.47         8
weighted avg       0.77      0.88      0.82         8


✅ Model saved as dysgraphia_model.pkl
PS C:\Users\Server\Documents\dysgraphia-detector\ml\pen_ml_model> python .\test_model.py 
✅ Model loaded successfully.
Batch 1: Normal
Batch 2: Normal
Batch 3: Normal
Batch 4: Normal
Batch 5: Normal
Batch 6: Normal
Batch 7: Normal
Batch 8: Normal
Batch 9: Normal
Batch 10: Normal
Batch 11: Normal
Batch 12: Normal
Batch 13: Normal
PS C:\Users\Server\Documents\dysgraphia-detector\ml\pen_ml_model> python .\test_model.py
✅ Model loaded successfully.
Batch 1: Normal
Batch 2: Normal
Batch 3: Normal
Batch 4: Normal
Batch 5: Normal
Batch 6: Normal
Batch 7: Normal
Batch 8: Normal
Batch 9: Normal
Batch 10: Normal
Batch 11: Normal
Batch 12: Normal
Batch 13: Dysgraphia
Batch 14: Dysgraphia
Batch 15: Dysgraphia
Batch 16: Normal
PS C:\Users\Server\Documents\dysgraphia-detector\ml\pen_ml_model>

~~~