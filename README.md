
Copyright of code and data is owned by respective authorities. Strictly for learning purpose


# Tensorflow_Chatbot
This repository contains the code for a general purpose chatbot built using the seq2seq model. It is a well commented code , hence all the required explanation has been  provided in good detail.
The chatbot can also be trained on other data as required.

The process was divided into 3 parts 
1. Data preprocessing
2. Building the seq2seq model
3. Training and Testing the model

1. Data Processing 
    The dataet being used is the cornell movie dialogues dataset. However due to limitation of size of file being uploaded I     am unable to include that. However its open source and can be fount at
    http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
 
  i. The first step is to map each input sentence to it's unique id which basicaly is the first element in each line.
  ii.The next step includes creating a huge list of conversations where we consider the id's of each sentence we obtained in       step i.
  iii.Our chatbot will be trained on questions and their answers so we create 2 different lists namely questions and               ansswers.
  iv. Now we clean both the lists.In this step we remove the unnecessary characters in the training data. eg:- !, !!, ...,         hmm, etc. and the words which occur less requently.
  v. Finally we create and append tokens to the cleaned answers list so that the seq2seq model will know the beginning and         end of each answer.We then decide the length of input sentences
  

2. Building the seq2seq model
  i. Create place holders for input and targets. Decide the learning rate and the drop out rate
  ii. As the decoder will accept only a specific format of I/P(In batches and each sentence starting with SOS token) we             preprocess the targets.
  
  iii. Generate Encoder RNN-  Now we build the  which is of the form LSTM. For encoder cells we use bidirectional dynamic Rnn         to obtain the encoder states.
  
  iv. The next step is to build the decoder RNN. It includes the following steps
  
       # 1 : decode the training set - Decoder gets the encoder states as input which is output by encoder Rnn. This step                  also implements the attention mechanism
             so that the decoder avoids printing the same word with highest probablity at that instant over and over again.
              
       # 2 : encode the validation  set - This step provides a logic to the chatbot based on which it predicts the output for              new questions. It generates a context vector during each iteration of training.
             It outputs the predicted word which iss forwarded in the next cell.
                                          
       # 3 : Generate the decoder RNN - This is also of the form LSTM. The core function here is the output_function whcih                 returns fully connected lstm layers.
      
      
   v. The seq2seq model is basically the combination of encoder RNN and decoder RNN. Now we assemble the seq2seq model
   
   
  3. Training and Testing the model
  i. We set the hyper parameters
  ii. Select the drop out rate.
  iii.Create an interactive session and obtain the training and test predictions using the seq2seq model we created.
  iv. We calculate loss error based on weighted cross entropy between the training predictions and the targets
  v.  We use the adam optimizer to calculate the gradient descent and then we also apply gradient clippping.
  vi. We apply padding and split the data into batches of questions and answers. Divide the data into training and test set         and start trainining.
  vii. We then set up a simlply interface to chat with the chatbot after training has been completed.
  
  

  
  
