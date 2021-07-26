import argparse, os
import pandas as pd
import boto3
import sagemaker
import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
#from keras import backend as K
#import tensorflow.python.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report



if __name__ == '__main__':
    
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=20)
    #parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--lstm-out', type=int, default=196)
    parser.add_argument('--batch-size', type=int, default=64)
    
    #sm_model_dir: model artifacts stored here after training
    #training directory has the data for the model
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))


    #Read hyperparams and data
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    #lr         = args.learning_rate
    embed_dim = args.embed_dim
    lstm_out = args.lstm_out
    batch_size = args.batch_size
    
    #model artifacts and data
    training_dir   = args.train


    #Data Reading & Preprocessing
    print("reading")
    df = pd.read_csv(training_dir + '/train.csv',sep=',')
    print("read")
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(df['text'].values)
    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X)
    Y = pd.get_dummies(df['sentiment']).values
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size = 0.20, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_val.shape,Y_val.shape)
    
    
    #pass in as hyperparams
    #model building
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.fit(X_train, Y_train, 
              epochs = epochs, 
              batch_size=batch_size,
              validation_data=(X_val, Y_val),
              verbose = 1)

    print("Saving model")

    #Replace number with model version do not repeat versions otherwise will run into an error
    model.save(os.path.join(args.sm_model_dir, '000000004'))
    
    
    
    
    
    
