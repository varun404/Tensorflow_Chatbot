import tensorflow as tf
import re
import numpy as np
import time


#The lines and their id's
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')



#The lines used in the form of conversation.
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


'''+++++++++++++++Dataset Explanation++++++++++++++'''
#In the dataset there are total of 5 columns.
#Column 1 - Dialogue Id
#Column 2 - U means user. Actor that is. So U0 then U2 then U0 actually represents the consequent dialogues between actor 0 and 2 in the movie_lines.txt
#Column 3 - M  means Movie. Represents Movie ID
#Column 4 - Actors name
#Column 5 - Actual Dialogues
'''-------------------------------------------------'''



#########################################
# -----------1 Data preprocessing---------#
#########################################

#We first map the sentences to their respective id's
#We store the result in dictionary id2line
id2line = {}
for line in lines:
    #_templine is a local variable which is local to the loop itself. 
    
    #If we look at the text file we can see that we basically need the first and last 
    #elements. So we can split at +++$+++ whichwill give us a total of 5 components in each entry in _templine
    #Out of the 5 we need the first and the last and store them in id2line dictionary
    _templine = line.split(' +++$+++ ')
    if((len(_templine)) == 5):
        id2line[_templine[0]] = _templine[4]
        


#We will be using the conversations as input to the neural network.
#So we just obtain a huge list of all the conversation id's
conversation_id = []
for conversation in conversations[:-1]:
    #Consider only the characters inside ['  '] and ignore the rest i.e ignore u0 ,us and so on.
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","")
    #_conversation = re.search("\'.+'",conversation)
    conversation_id.append(_conversation.split(','))
    

#Basically our chatbot will be trained on questions and answers. So we make a seperate list of question and a seperate one for answers
#This will be the input to the nerural network.
#So here we basically prepare the input to the neural network
#In conversation_id each alternate ie. in each line , each element alternately is question and the answer.

#Basically we will have 2 lists i.e questions and answers where question at a line number in the question list will have an answer at the same id in the answer list
questions = []
answers = []

#Go through every line in conversation_id
for conversation in conversation_id:
    #in each line which will be a list get the number of elements - 1
    #For eg for i = 0 we will have len(conversation = 4)
    #But as we are considering 2 elements at a time we dont have to iterate over complete len of conversation
    for i in range(len(conversation) - 1):
        #Now as we know the elements will be of the form[question , answer, question ,...]
        #So in questions list we append first element i.e, i
        #In answer list we append the next element i.e i+1
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
        

#After splitting the original data into subsections we have to clean the data.

#Preprocessing        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-a()\"#/@;:<>{}+=-|.?,]", "", text)
    return text                  




##Now we clean the questions. Course method. Uses extra list

clean_questions = []
for questn in questions:
    cl_q = clean_text(questn)
    clean_questions.append(cl_q)

#Now  we clean answers.Course method. Uses extra list
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


#-----------------------------------------------



#Now we can eliminate the words which are used rarely. That is their count is low
#So first step will obviously will be to get the count of each word.
word_count = {}
for question in clean_questions:
    for word in question.split():
        if word in word_count:
            word_count[word]+=1
        else:
            word_count[word] = 1
  
            
for answer in clean_answers:
    for word in answer.split():
        if word in word_count:
            word_count[word]+=1
        else:
            word_count[word] = 1  
            
            
                        
#Now we start eliminating words based on their frequency. A good way to do this would be to consider tokenization.
#We can also keep the tokens list in a sorted order to see the count of words in ascending order. Basically we manually assign the index in ascending order.

#To elimante a word it's count should be less than a threshold frequency. here we can consider the threshold count to be 10
threshold= 15           
manual_index = 0

questions_count = {}
for word, count in word_count.items():
    if word in ["!", "!!!", "!!", "...", "(", ")", "\", """, "#", "/", "@", ";", ":", "<", ">", "{", "}", "+", "=", "-", "|", ".", "?", ","] or count < threshold:
                pass
                #word_count.popitem()
    else:
        questions_count[manual_index] = word
        manual_index+=1
        

answer_count = {}
for word, count in word_count.items():
    if word in ["!", "!!!", "!!", "...", "(", ")", "\", """, "#", "/", "@", ";", ":", "<", ">", "{", "}", "+", "=", "-", "|", ".", "?", ",", "]"] or count < threshold:
                pass
    else:
        answer_count[word] = manual_index
        manual_index+=1
        
        
#Now we create the SOS token and EOS token so that the seq2seq models at the encoding part will know where to start and end
#We also create the OUT token which will act as a trigger. If a word's frequency is less than threshold then it'll be repaced by OUT and seq2seq model will not consider the word.
#We also create the PAD token to implement padding as the input to the Encoder should always be of same length.
        #Eg: all the sequences in your batch should have the same length. If the max length of your sequence is 8,
        #your sentence My name is guotong1988 will be padded from either side to fit this length: My name is guotong1988 _pad_ _pad_ _pad_ _pad_

tokens = ['<PAD>' , '<EOS<' , '<OUT<' , '<SOS>']

#We have to do this for both questions and answers as both are provided as I/P
for token in tokens:
    questions_count[token] = len(questions_count) + 1

for token in tokens:
    answer_count[token] = len(answer_count) + 1
    
    
#We create the inverse dictionary of answer_count to map the countof the word in answer.
inverted_answer = {value:key for key , value in answer_count.items()}
    

#Now we have out I/P compatible to the  Encoder. Now we focus on the Decoder which will O/P answer.
#But in order to identiy the end of an O/P i.e a single answer we use the EOF token.
#So we loop through the answers list we already have.
        
for i in range(len(answers)):
    #Note that we added a <space> before <EOS> in order to differentiate between adjacent words
    clean_answers[i] += ' <EOS>'
    
    
    
    
#Now one thing we should remember that we do not provide text input to the seq2seq model. We have to provide a vector(list here i.e; array) of numbers.
#Each element in the list here will basically contain an array of numbers. Each number will represent the word in the question.But instead of the word
#We use the unique integer assigned to it in the questions_count list

#WE DO THIS TO MAKE SURE THE  LENGTH OF INPUT IS THE SAME EVERYTIME
questions_integer = []
for question in questions:
    #We use the following list as a temporary(local) buffer.
    ints = []
    for word in question.split():
        #Now we first check if the word is present in the questions_count list. If it's not then it means that its an OUT token.
        if word not in questions_count:
            pass
            #####ints.append(questions_count["<OUT>"])
            #If it is present then it means 
        else:
            ints.append(questions_count[word])    
            questions_integer.append(ints)
        
        
        
        
answers_integer = []
for answer in answers:
    #We use the following list as a temporary(local) buffer.
    ints = []
    for word in answer.split():
        #Now we first check if the word is present in the questions_count list. If it's not then it means that its an OUT token.
        if word not in answer_count:
            pass
            #####ints.append(answer_count['<OUT>'])
            #If it is present then it means 
        else:
            ints.append(answer_count[word])
            answers_integer.append(ints)
            
            
#Next step is to limit the length of I/P.
#We decide the length based on questions as it'll reduce the padding during training
sorted_length_questions = []
sorted_length_answers = []

#We decide the length of the sentence to be 50 i.e 28 words in a sentence
for length in range(1 , 28):
    #We basically need the question so that we can iterate over it. We can refer the question by the index of the sentence.
    #We basically form a tuple of the index of the question and the whole sentence.
    for i in enumerate(questions_integer):
        if len(i[1]) == length:
            #Note that i will be a tuple of index and question so length of i will be 2. i[0] will be index and i[1] is the whole question.
            sorted_length_questions.append(questions_integer[i[0]])
            sorted_length_answers.append(answers_integer[i[0]])
 
           
#################################################
# -----------2 Building the seq2seq model---------#
#################################################
            
            
#Step 1 : Creating placeholders for the inputs and the targets
def model_inputs():
                            #As our input type is integers, we select the data type as int32
                                    #2nd argument is the dimension of I/P data. Including the padding we have 2 dimensional input i.e sorted_length_questions
                                                       #Name for the input             
    inputs = tf.placeholder(tf.int32 , [None , None] , name = 'input')
    
    #Targets are the answers.
    targets = tf.placeholder(tf.int32 , [None , None] , name = 'target')
    
    #Setting learning rate
    lr = tf.placeholder(tf.float32 , name = 'learning_rate')
    
    
    #Keep prob:
    #It controls the dropout rate while training the NN.
    
    #Drop out rate :
    #The keep_prob value is used to control the dropout rate used when training the neural network.
    #Essentially, it means that each connection between layers will only be used with probability 0.5 when training. 
    #This reduces overfitting as with probablity of each neuroon being used at an instant is 0.5.
    #In other words during each iteration of training the number of active neurons will vary. It'll reduce or increase.
    
    #It means the weights of certain % of neurons will not be upadated . This rate at which the neurons are made inactive is called as dropout rate.
    keep_prob = tf.placeholder(tf.float32 , name = 'keep_prob')
    
    return inputs , targets , lr , keep_prob
######################################################

#Step 2 : Preprocess the targets.
#The decoder will only accept certain format of targets.
#The format is 2 fold:
        # 1) - Targets must be provided in batches.
        # 2) - Each answers in the batch must start with SOS tokens.
              # So we'll have to add the SOS token . But the input size must be consistent. Adding of additional token will vary the size.
              #So we can delete the last comlumn and thus concatenate the SOS token at the beginning.
              
def preprocess_targets(targets , word_count , batch_size):
    
                        #Since we have to obtain the integer value of SOS token we include the word_count dictionary.
    #Left_side is basically a matrix or a vector of batch size lines one column which will contain the SOS token.
    
            #Here we basically just make a matrix by using the tf.fill() which takes 2 inputs
                        #In [batch_size , 1] is a tensor and 1 is the dimensional value of it.
                                          #word_count['<SOS>'] denotes the value of the token to be filled in the matrix
    left_side = tf.fill([batch_size , 1], word_count['<SOS>'] )                    
    
    #Now we make the right side which is all the answers in the batch except the last column
    #tf.strided_slice() is used to do numpy style slicing of a tensor variable. It has 4 parameters in general: input, begin, end, strides.
    #The slice continues by adding stride to the begin index until all dimensions are not less than the end
                                 #input    #begin  #end- -1 means except last column
                                                                     #This represents how many number of cells it should slide during extraction. 
                                                                      #Since  we want all the rows and all the columns we input [1 , 1]  i.e: each cell. Moving one cell at a time         
    right_side = tf.strided_slice(targets, [0][0], [batch_size, -1], [1 , 1])
    
    #Concatenate takes 2 inputs.
                                     #values: which is a tuple which will be concatenated
                                                               #Axis: Horizontal or Vertical concatenation
                                                                       #1         #0
    preprocessed_targets = tf.concat([left_side , right_side] , 1)
    return preprocessed_targets  

############################################################

#Step 3 : Building Encoder Part(LSTM)     
                    #input will be the values returned by model_inputs()
                                 #rnn_size will be the number of neuron in input layer
                                                       #As we are going to drop out regularisation to obtain stack lstm. Better efficiency
                                                                    #List of length of each question in a batch.
def encoder_rnn(rnn_input , rnn_size, num_layers, keep_prob, sequence_length):
    #First we make an object of the Basic LSTM  Class.
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    #Now we apply dropout to the lstm
    #Droput deactivates the % of neurons i.e dosent update the weights of the neurons
    #Inputs : The RNN, 
    #         dropout rate i.e keep_prob. Because keep_prob controls the dropout rate.       
                                                #This is the previous lstm layer given as input i.e it's basically an object of previous lstm layer being passed on
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    
    #Now we create the Encoder Cell
    #Inputs : 1) Number of layers. i.e: We apply dropout on every lstm layer. And we take num_layers as I/p.
    #         So the total number of cells to be given as I/P will be (number of dropout neurons) * Total number of layers
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    #We now create a dynamic bidirectional RNN.
    #Dynamic bidirectional RNN generates the Forward and backward RNN based on the Input provided
    #While constructing a bidirectional RNN the I/P size of the forward and backward rnn must be same.
    '''Note : bidirectional_dynamic_rnn() returns 2 outputs
    i.e encoder_output and encpder_state. We only need 
    encoder_state so we can replace the first parameter by _
     |
     |
    \ /
     |  '''
    _, encoder_state  = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                        cell_bw = encoder_cell, 
                                                        sequence_length = sequence_length, 
                                                        inputs = rnn_input, 
                                                        dtype = tf.float32) 
    return encoder_state
###########################################################
    

#Step 4 : Building the decoder.
       
   # 4.1 : decode the training set
   # 4.2 : encode the validation  set
   # 4.3 : Generate the decoder layer(Actual RNN)

#4.1  
                       #Decoder gets the encoder states as input which was output by encoder_rnn().
                                       #The cell in rnn of decoder.
                                                   #The inputs to which we apply embeddings.
                                                   #Embedding: Mapping words to vectors of 
                                                   #real numbers. Each dimension(element) 
                                                   #in the vector is not related to other
                                                   #dimension. It is just a measure of the
                                                   #location and distance between 2 
                                                   #neighbouring vectors that is used as
                                                   #basis for training.
                                                                                              #tf.variable_scope(): 
                                                                                              #It is a special DS 
                                                                                              #that wraps tensorflow variables.
                                                                                              #A context manager for defining ops
                                                                                              #that creates variables (layers).
                                                                                              #This context manager validates that
                                                                                              #the values are from the same graph,
                                                                                              #ensures that graph is the default graph, 
                                                                                              #and pushes a name scope and a variable scope.
                                                                                              ##If the same name has been previously used in
                                                                                              #the same scope, it will be made unique 
                                                                                              #by appending _N to it.
                                                                                                              
                                                                                                              #It is used to return decoder output
def decoder_training_set(encoder_state, decoder_cell, decoder_embedded_inputs, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    #We first have to get the attention states. For that we first have to initialize them as 3-D matrix of 0's
    #Inputs : 
    #1 no. of lines = Lines means observations here i.e the lines of text. Since we are working in batches the number of lines will be batch_size
    #2 no. of columns = 1 
    #3 no. of elements in the 3rd Dimension i.e 3rd axis = decoder cells output size
            
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    #Now next step is to create keys, values, functions(score function, construct function) for attention mechanism.
    #Attention keys are to be compared with target states.
                    #Attention values are used to construct context vector
                    #which is the first element of decoding
                                       #Attention_score is used to compute similarity between 
                                       #keys and target states
                                                                 #attention_construct_function is used to construct attention state.
    attention_keys , attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
                                                                              #To obtain the value of the state                                                                           #To define a namescope 
                                                                                                                                                                                          #If name_or_scope is not None, it is used as is. 
                                                                                                                                                                                          #If name_or_scope is None, then default_name is used. 
                                                                                                                                                                                          #In that case, if the same name has been previously used in the same scope, it will be made unique by appending _N(integer value) to it.
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name = "attn_dec_train")
    
    #Finally we get the decoder output
    #decoder_output, decoder_final_state, decoder_final_context_state are the values returned by the decoder output function  of tf
    #But we only need decoder_output. So we can replace the other 2 parameters by _
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_inputs, sequence_length, scope = decoding_scope)
    #             (2.decoder_final_state
     #              3. decoder_final_context_size  ) 
    #The last step would be to apply a dropout to decoder output whilst keep_prob keeping a track of the dropout_rat
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

#########################################################################
    
#Step 5 : Decoding the test set
#Training set's values  are propagated back in the NN that will help update the weights.

#Test set is used for validation(cross validation here)
    
#In decoding training_set we used attention_decoder_fn_train() to decode the training set
#Here in test set we use attention_decoder_fn_inference()
#Inference means to deduce logically which makes sense. As our chatbot learns the logic during training.
#So the chatbot can use the logic to predict answers for a new question.


#The arguments will be arguments of the attention_decoder_fn_train() and 4 more
#1 sos_id - sos token id
#2 eos_id  - sos token id
#3 maximum_length - Length of longest answer in the batch.
#4 num_words  - Total number of words of all answers. Consider the length of answers_integer list.

                                            #Which gives the decoder_embedded_inputs
def decoder_test_set(encoder_state, decoder_cell, decoder_embedded_matrix, sos_id, eos_id, maximum_length, num_words,  sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    #Now next step is to create keys, values, functions(score function, construct function) for attention mechanism.
    #Attention keys are to be compared with target states.
                    #Attention values are used to construct context vector
                    #which is the first element of decoding
                                       #Attention_score is used to compute similarity between 
                                       #keys and target states
                                                                 #attention_construct_function is used to construct attention state.
    attention_keys , attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
                                                                              #To obtain the value of the state                                                                                           #To define a namescope 
                                                                                                                                                                                                          #If name_or_scope is not None, it is used as is. 
                                                                                                                                                                                                          #If name_or_scope is None, then default_name is used. 
                                                                              #Because validation set will not return the values to be back propagated in the NN
                                                                              #What that means is that the output predictions will not be used in modifying weights
                                                                              #So instead of returning the output_function, we provide it as an I/P arg  .                                                 #In that case, if the same name has been previously used in the same scope, it will be made unique by appending _N(integer value) to it.
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function,
                                                                              decoder_embedded_matrix, sos_id, eos_id, maximum_length, num_words, #Additional parameters
                                                                              name = "attn_dec_inf")
    
    #Finally we get the decoder output
    #decoder_output, decoder_final_state, decoder_final_context_state are the values returned by the decoder output function  of tf
    #But we only need decoder_output. So we can replace the other 2 parameters by _
    
    test_predictions  = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, test_decoder_function,  scope = decoding_scope)
    #We wont be providing any weight modoification values. So it'll be fine to ignore decoder_output, _, _.
    

    return test_predictions    

  #########################################################################

#Step 6 : Building the Decoder Rnn  
  #We make the decoder rnn function that builds the rnn of the decoder
  
                                                                    #Encoder state is the O/p of encoder
def decoder_rnn(decoder_embedded_inputs, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word_count, keep_prob, batch_size):
    #We first define the decoding scope by giving a name to the variable scopeii i.e "decoding"  here.
    with tf.variable_scope("decoding") as decoding_scope:
        #Now we first create the lstm layer
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        
        #Now we apply dropout to each layer which eventually is applied to each neuron.
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        
        #But we are making a stacked rnn.Which basically means that we'll have multiple lstm layers
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        
        #Now we inititalize the weights.
        #We use truncated_normal_initializer() because it selects mean and the values which are close to zero
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        
        #Now we initialize the biases.
        biases = tf.zeros_initializer()
        
        #Now we make the output function which will return  fully_connected layer.
        #Fully_connected layer is present only at the end of the rnn
        #lstm cells comes first and then fully connected layer comes at the end.
        #We already have the decoder_cell composed of stacked lstm layers with dropout applied to each layer
        #But essentially we have to make the fully connected layer
        output_function = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                      num_words,#The length of answer
                                                                      None, #We can select the activation function and regularization techniques. But we go with the default Relu activation and no regularization. So No modifications to the default.Hence None
                                                                      scope = decoding_scope, 
                                                                      weights_initializer = weights, 
                                                                      biases_initializer = biases)
        
        training_predictions = decoder_training_set(encoder_state, decoder_cell, decoder_embedded_inputs, sequence_length, decoding_scope, output_function, keep_prob, batch_size)
        
        
        #Now we can generate the test predictions for cross validation with training set.
        #But we have already defined a deccoding scope in the above statements as "decoding"
        #So in order to specify a new scope i.e we basically reuse the variables
        decoding_scope.reuse_variables()
        test_predictions = decoder_test_set(encoder_state, decoder_cell, decoder_embedded_matrix, word_count['<SOS>'],word_count['<EOS>'], sequence_length - 1, num_words, decoding_scope, output_function, keep_prob, batch_size)
                                                                                                  #SOS token id       #EOS token id
                                                                                                  
    
        return training_predictions, test_predictions
    
###################################################################################################
        
#Step 7 : Building the seq2seq model
    #This is the combination of encoder_rnn and decoder_rnn
    #It should return training predictions and test predictions.
    #We meed encoder_rnn because it returns encoer states which is used as input to the decoder to give test predictions as O/P
    
                  #Questions of dataset(before training)
                  #Questions asked by user(after training)
                          #The real answers from the dataset                
                                                                             #Total number of words in all answers
                                                                             #Similar to the words_count dictionary
                                                                             #but this contains only the total number
                                                                             #of answers.
                                                                                                #Total number of words in all questions
                                                                                                #Similar to the words_count dictionary
                                                                                                #but this contains only the total number
                                                                                                #of questions.       
                                                                                                                     #Number of dimension of embeddding matrix of the encoder.
                                                                                                                                             #Number of dimension of embeddding matrix of the decoder.   
                                                                                                                                                                                           #To preprocess the targets
def seq2seq_model(inputs, targets, keep_prob, batch_size,  sequence_length,  answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questions_integer):
                                            
                                                #This function returns embedded inputs of the encoder.
    encoder_embedded_inputs = tf.contrib.layers.embed_sequence(inputs, #What we want to embed
                                                               answers_num_words + 1, #Total number of words of answers. +1 is used as upper bound is excluded
                                                               encoder_embedding_size, 
                                                               initializer = tf.random_uniform_initializer(0, 1))#Initialize initial weights randomly. Values should be between 0 and 1.
    
    #Now we get the output of the encoder (from encoder_rnn we made) which will be fed as input to the decoder
    encoder_state = encoder_rnn(encoder_embedded_inputs, rnn_size, num_layers, keep_prob, sequence_length)
    
    #Finaly we can get the training and test predictions
    #But first we need to get the preprocessed targets because we need them for training
    #We also need to get the embedding_matrix which is used to get decoder_embedded_inputs
    
    preprocessed_targets = preprocess_targets(targets, questions_integer, batch_size)
    
    #Embedding matrix
    #Variable() takes arguments which are basically the dimension of the matrix.
    #Embedding matrix will be continuously updated during training but initially we fill it with random numbers(refer Andrew Ng Embedding Matrix)
    
    
    
    #The number of rows will obviously be the number of questions and the number of columns will depend on the embedding size.
    #i.e how many one e should be obtained at a time(Refer Andrew Ng first)                                  #lower bound and upper bound of random numbers is 0 and 1 resp.  
    decoder_embedding_matrix = tf.Variable(tf.random_uniform([questions_count + 1, decoder_embedding_size]), 0, 1)
    
    #Now as we have obtained the embedding matrix we can obtain the inputs to the decoder
    decoder_embedded_inputs = tf.nn.embedding_lookup(decoder_embedding_matrix,#embedded questions
                                                     preprocessed_targets,)#They are basically the answers to the questions 
    
    #As the main aim of seq2seq is to return training and test predictions we have to obtain that now.
    
    #This will be propagated back in the rnn.
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_inputs, 
                                                         decoder_embedding_matrix, 
                                                         encoder_state, 
                                                         questions_num_words, 
                                                         sequence_length, 
                                                         rnn_size, 
                                                         num_layers, 
                                                         questions_integer,
                                                         keep_prob, 
                                                         batch_size)
    return training_predictions, test_predictions
    
#####################################################################################



#################################################
# -----------3 Training the seq2seq model---------#
#################################################

#Step 1  : Setting the Hyperparameters
#Epochs = Getting inputs as per batch size in the nn 
          #forward propogating them inside the encoder to get encoder state
          #forward propogate the encoder state along with target into decoder to get the final o/p score and then the o/p answer
          #backpropogate the answer scores and the loss in the nn
          #update the weights.
epochs = 80
batch_size = 100
rnn_size = 512
num_layers = 5#includes encoder and decoder layers
encoder_embedding_size = 512  #number of columns in embedding matrix. here we wil have 512 columns in our embedding matrix
decoder_embedding_size = 512
learning_rate = 0.018
learning_rate_decay = 0.8  # % by which learning rate is reduced during each iteration so that the model can understand the correlation between questions and answers better Here we can try 80%
min_learning_rate = 0.0001


#Keep prob(keep probablity)
#Basically a neuron being active witout dropout rate applied is 100% i.e 1
#When we apply a dropout rate p the probablity of it being active becomes 1 - p
#what that means the neuron will be deactivated with probablity 1- p 
#But during test all the neurons should be present
#So we do not use it for obtaining test predictions.

#Dropout rate  are applied on input layer as well as hidden layers.
#Here we've applied it on the hidden layer.
# The optimal  values for dropout rate is 20% for input unit and 50% for hidden units

keep_probability = 0.5 #Used only during training obviously for droput rate during training


#Creating session to train seq2seq model
#Reset tensorflow graphs first
#reset_default_graph() : Clears the default graph stack and resets the global default graph

tf.reset_default_graph()

#Now create the sesssion
session = tf.InteractiveSession()


######################################################################################

#Loading the model inputs by calling the model_inputs()
#as it returns inputs, targets, learning rate and keep_prob hyperparameters
inputs, targets, lr, keep_prob = model_inputs()


#Setting the sequence length.
#We consider maximum length to be 28
#We set this parameter because encoder_rnn() as well as the decoder_rnn() has this as its parameter
#We can set a default value but it should be a tensorflow placeholder(variable)
                                             #Maximum number of words
                                             #in a sentence during training
                                             
                                                 #The dimensions of the tensor input
                                                 #sequence_length is just a variable 
                                                 #and not a list of integers. So we put None
sequence_length = tf.placeholder_with_default(27, None, name = "sequence_length")



#Now we get the shape of the inputs tensor
#Our inputs are in in the form of the tensor
#we need to get the shape of the tensors becauss
#it will be used as an argument of a specific function 
#The specific function is actually ones() which will create a 
#tensor of ones and the dimensions will be exactly equal to the inputs shape
input_shape = tf.shape(inputs)

###################################################################################
#Getting the training and test predictions(Not training)
#This part actually emphasis on the training and test predictions which we will get after actually specifying
# all of the inputs, targets, lr, keep_prob parameters
#We obviously do this using the seq2seq model we created
#They will be used later on during training                #Refer documentation
training_predictions , test_predictions = seq2seq_model(tf.reverse(inputs, [-1]), 
                                                        targets, 
                                                        keep_prob, 
                                                        batch_size, 
                                                        sequence_length, 
                                                        len(answers_integer), 
                                                        len(questions_integer), 
                                                        encoder_embedding_size, 
                                                        decoder_embedding_size, 
                                                        rnn_size, 
                                                        num_layers, 
                                                        questions_count)

####################################################################################




#Setting up the loss Error, the optimizer and gradient clipping
#Gradient Clipping : Gradient clipping is most common in recurrent neural networks.
#                    When gradients are being propagated back in time, they can vanish because they they are continuously multiplied by numbers less than one. 
#                    This is called the vanishing gradient problem. This is solved by LSTMs and GRUs, and if you’re using a deep feedforward network, this is solved by residual connections. 
#                    On the other hand, you can have exploding gradients too. This is when they get exponentially large from being multiplied by numbers larger than 1. 
#                    Gradient clipping will clip the gradients between two numbers to prevent them from getting too large


#Loss Error :  (Based on weighted cross entropy) : Optimal when working with sequences
#              weighted_cross_entropy_with_logits is the weighted variant of sigmoid_cross_entropy_with_logits. 
#              Sigmoid cross entropy is typically used for binary classification. Yes, it can handle multiple labels, but sigmoid cross entropy basically makes a (binary) decision on each of them 
#              For EXAMPLE :- for a face recognition net, those (not mutually exclusive) labels could be "Does the subject wear glasses?", "Is the subject female?", etc.


#Optimizer : First we will use adam optimizer for stocahstic gradient descent and then apply gradient clippping to it to avoid vanishing or exploding gradient


with tf.name_scope("Optimisation"):
                                    #This measures loss between training_predictions and targets
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, 
                                                  targets, 
                                                  tf.ones([input_shape[0], sequence_length]))#This will create the tensor of voids and fill it with ones where number of rows = input_shape and the index column only so we use [0] and columns = sequence_length
    #Now we prepare the optimizer
    # 1 Get the adam optimizer.
    # 2 Prepare gradient clipping i.e cap the gradients between minimum value and maximum value
    # 3 Apply it to our optimier
    
    # 1 Getting the adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # 2 Clip all our gradient
    # We have one gradient per neuron in a nn.
    # and for each neuron we compute the gradient of the loss error 
    # w.r.t the weight of the neuron.
    # All gradients are in a graph and attached to a variable
    
    # First we compute all the gradients
    gradients = optimizer.compute_gradients(loss_error)
    
    # 3 Now we cap each gradient
    # As we have to cap each gradient in the graph we use for loop
    clipped_gradient = [(tf.clip_by_value(grad_tensor, -5., 5), grad_variable )for grad_tensor, grad_variable in gradients if grad_tensor is not None]
                                                      #min #max
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradient)                                                      
    
    
#######################################################################################################
    
#Padding the sequences with <PAD> token
# Question : {'Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>}
# Answer : {'<SOS>', 'I', 'am', 'a', 'bot', '.',  <EOS>}    

def apply_padding(batch_of_sequences, word_count):
    #We first find the sentence with maximum  number of words in a batch and pad other sentences accordingly
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word_count['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

########################################################################################################

#Splitting the data into batches of questions and answers
def split_into_batches(questions , answers, batch_size):
        for batch_index in range(0, len(questions) // batch_size):
                    #index of first question being added to the batch
                    start_index = batch_index * batch_size
                    questions_in_batch = questions[start_index : start_index + batch_size]
                    answers_in_batch = answers[start_index : start_index + batch_size]
                    padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questions_integer))
                    padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answers_integer))
                    yield padded_questions_in_batch, padded_answers_in_batch
                    
############################################################################################################


# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_length_questions) * 0.15)
training_questions = sorted_length_questions[training_validation_split:]
training_answers = sorted_length_answers[training_validation_split:]
validation_questions = sorted_length_questions[:training_validation_split]
validation_answers = sorted_length_answers[:training_validation_split]
                     
        
############################################################################################################            


##################
# Actual training
##################

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            
            
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
    
    
                total_validation_loss_error += batch_validation_loss_error
                
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            
            
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
                
            list_validation_loss_error.append(average_validation_loss_error)
            
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
                
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
    
print("Game Over")

   
####################################################################################################################


#Now we have to load the weights and run the session
checkpoint - "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

#We connect loaded weights to the session
saver = tf.train.saver()
#connect session to the checkpoint
saver.restore(session, checkpoint)

#Now we have to converst the questions from strings to list of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
                       #If word is not in dictionary then consider the out token
    return [word_count.get(word, word_count['<OUT>']) for word in question.split()]

#######################################################################################################################


#Select the following and execute to chat with the chatbot.
    
#Finally we can set up the chat window.
while(True):
    question = input("You : ")
    if question == 'Goodbye':
        break
    question = convert_string2int(questions, questions_integer)
    
    #First we implement padding to obtain a consistent length
    question = question + [questions_integer['<PAD>']] * (27 - len(question))
    fake_batch = np.zeros((batch_size, 27))
    
    #Now we have to include our question into the fake batch
    fake_batch[0] = question
    
    #Now we can feed it into nn to get predicted answer of the chatbot
    predicted_answer = session.run(test_predictions, {inputs : fake_batch, keep_prob: 0.5})[0]
    
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answers_integer[i] == 'i':
            token = 'i'
        elif answers_integer[i]  == '<EOS>':
            token = '.'
        elif answers_integer[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answers_integer[i]
        answer += token
        if token == '.':
            break
    print('Chatbot : '+ answer)
        
        
    
    
                        

    
    


      




    
    
    



    
                
                        
                                
    
    
    
    
    
    
            


    



        


        
        
        
        
        
        
        
        
        


    