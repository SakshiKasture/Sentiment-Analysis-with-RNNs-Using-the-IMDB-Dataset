IMDB Movie Reviews Sentiment Analysis with RNN
📄 Project Overview
This project demonstrates sentiment analysis on the IMDB movie reviews dataset using a Recurrent Neural Network (RNN). The model predicts whether a given movie review expresses positive or negative sentiment. The implementation includes proper data splitting for training, validation, and testing to ensure robust performance evaluation.

📊 ***Dataset Used
IMDB Movie Reviews Dataset
Contains 25,000 labeled movie reviews.
Labels: Positive (1) or Negative (0).
Reviews are encoded as sequences of integers, where each integer represents a specific word from a dictionary of the top 10,000 most frequent words.
🛠️ ***Technologies Used
Python: Core programming language for implementation.
TensorFlow/Keras: For building and training the RNN model.
NumPy: For numerical operations.
Scikit-learn: For splitting the dataset into training, validation, and testing sets.

📝 ***Project Details and Code Explanation
1. Data Preprocessing
Loaded the IMDB dataset with the top 10,000 most frequent words.
Padded the sequences to ensure uniform input length (maxlen = 200).
Split the dataset into:
Training Set: For model training (60% of the data).
Validation Set: For monitoring performance during training (20%).
Test Set: For final evaluation (20%).
2. Model Architecture
Embedding Layer: Converts word indices to dense vector representations (word embeddings).
SimpleRNN Layer: Processes sequential data, capturing temporal dependencies in reviews.
Dense Layer: Outputs a single value (sentiment score) using the sigmoid activation function.
3. Model Compilation
Optimizer: Adam (adaptive learning rate optimization).
Loss Function: binary_crossentropy (for binary classification).
Metrics: accuracy.
4. Training and Validation
The model is trained for 5 epochs with a batch size of 64.
Validation data is used during training to monitor performance.
5. Evaluation
The model's accuracy is evaluated on the test dataset, which was not used during training or validation.

📌 ***Key Takeaways
Proper data splitting is essential for unbiased model evaluation.
RNNs are effective for processing sequential data, such as text reviews.
Validation data helps monitor overfitting during training.
🚀 ***Future Improvements
Experiment with more advanced architectures like LSTMs or GRUs.
Include pre-trained word embeddings like GloVe or Word2Vec for improved accuracy.
Perform hyperparameter tuning for further optimization.
