from common  import *
from array   import array
from tkinter import Y
import config
import logging,traceback
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.keras.layers import Input,MaxPooling1D, Convolution1D, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
#from sklearn.preprocessing import OneHotEncoder
from config import setting,log

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import theano 
import theano.tensor as T 

class LSTMCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs=None):
    log.info("LSTMCallback : on_train_begin(...)")
    #print("..LSTMCallback : on_train_begin(...)")
    
  def on_train_end(self, logs=None):
    log.info("..LSTMCallback : on_train_end(...)")
    #print('..on_train_end')
    global training_finished
    training_finished = True

earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

def onehot_poiID(onehot_duration):
  ## [0,0,0,1,0,0,0,0,12345]
  onehot=onehot_duration[:-1]
  duration=onehot_duration[-1]
  n=len(onehot_duration)-1

  for poiID in range(n):
    log.debug("-- onehot_poiID: onehot: %s", str(onehot) )
    log.debug("-- onehot_poiID: poiID: %d", poiID)
    log.debug("-- onehot_poiID: onehot_duration[poiID]: %s", str(onehot_duration[poiID]))
    if onehot_duration[poiID]==1:
      return( poiID,n,duration)
  else:
    log.error("-- ERROR: onehot_poiID('%s')", onehot_duration)

def lstm_model(X_input,y_output,y_reg, _input_size=2, _epochs=10, _dropout=0.3, _verbose_level=1):
  #print("# lstm_ model: type(X_input)=> ", type(X_input))
  #print("# lstm_model: type(Y_input)=> ", type(y_output))

  X = np.array(X_input)
  y_class=np.array(y_output)
  #y_class=y_output

  ###### SETTING
  # _verbose_level=2
  _lstm_units = len(X[0][0])-1 
  #_lstm_units = len(X[0][0])

  _batch_size = len(y_class)

  if setting["acticaton_function"]:
    _activation=setting["acticaton_function"]
  else:
    _activation="relu"
    setting["acticaton_function"]="relu"
  if False: ### Activation function
    pass
    # relu
    # tanh
    # softmax
    # sigmoid
    # leaky_relu

  #if setting["loss_function"]:
  #  _loss=setting["loss_function"]
  #else:
  ##  _loss="mse"
  #  setting["loss_function"]="mse"

  if False: ### _loss_class
    pass

  #_loss_function='MeanSquaredError'
  #_loss_function='kl_divergence'

  if False: ## _loss_function
    pass
    #_loss_function='mean_squared_logarithmic_error'

  _optimizer = None
  if setting["optimizer"]:
    _optimizer=setting["optimizer"]
  else:
    setting["optimizer"]="Adam"

  #print("# X.shape : ", X.shape)
  #print("# y_class : {", y_class, "}")
  #print("# y_class : {", y_class, "}")

  print("==> lstm.py: LINE_171 y_class = y_class.reshape( _batch_size:_{}_, 1, _lstm_units:_{}_)".format(_batch_size,_lstm_units))

  y_class = y_class.reshape(_batch_size, 1, _lstm_units)

  model_input = Input( shape=(_input_size, _lstm_units+1) )

  conv_net = Convolution1D( filters=7, kernel_size=3, activation=_activation, padding='same')(model_input)
  maxpool = MaxPooling1D(pool_size=2)(conv_net)

  ### LSTM model
  _lstm_model = LSTM( _lstm_units, return_sequences='false')(maxpool)
  _dropout = Dropout (_dropout)(_lstm_model)

  out_reg = Dense(1, activation='linear')(_dropout)
  out_class = Dense( _lstm_units, activation=_activation  )(_dropout)

  ### model
  _model = Model(inputs=model_input, outputs=[out_reg, out_class] )
  _model.compile( loss=[ setting['loss_function']], optimizer=_optimizer)

  ### DEBUG
  ### DEBUG
  ### DEBUG
  log.info("-- lstm.py : PRE-TRAINING: lstm_model(X_input,y_output,y_reg,_input_size=%d, _epochs=%d, _relu=%f, _dropout=%f, verbose_level=%d)")

  print("# LINE_217 lstm.py : PRE-TRAINING...")
  print("# LINE_218 lstm.py : PRE-TRAINING: {}"               .format(X.shape))
  print("# LINE_219 lstm.py : PRE-TRAINING: y_reg size : {}"  .format(len(y_reg)))
  print("# LINE_220 lstm.py : PRE-TRAINING: y_class size : {}".format(len(y_class)))

  ### training
  history_data = _model.fit(X, [y_reg, y_class], epochs=_epochs, batch_size=3, verbose=_verbose_level,\
    callbacks=[LSTMCallback()] )

  print("# lstm.py : FINSHED TRAINING.\n")
  return _model

####
class LSTMLayer(object):
  def __init__(self,X,dim,**kwargs):
    """
    Set up the weight matrices for a long short term memory (LSTM) unit. 
    I use the notation from Graves. 
    args:
        - dim: A dictionary containing the dimensions of the units inside the LSTM.  
    kwargs:
        - 
    """
    uni = np.random.uniform

    def diag_constructor(limit,size,n):
      """
      args:
          - limit: A list whose two elements correspond to the limit for the numpy uniform function.
          - size: (Int) one dimension of the square matrix.
          - n: The number of these matrices to create.
      """

      diag_ind = np.diag_indices(size)
      mat = np.zeros((n,size,size))
      for i in xrange(n):
        diag_val = uni(limit[0], limit[1],size)
        mat[i,diag_ind] = diag_val
      return mat.astype(theano.config.floatX)          

    truncate = kwargs.get("bptt_truncate", -1)

    nin = dim.get('in_dim')
    nout = dim.get('out_dim')
    nhid = dim.get('hid_dim')
    self.nin = nin
    self.nout = nout 
    self.nhid = nhid 
    # print("hidden dim", nhid)
    # I can cast weight matrices differently. Instead of creating separate weight matrices for each connection, I create them 
    # based on their size. This cleans up the code and potentially makes things more efficient. I will say that it makes 
    # the recurrent step function harder to read.
    self.Wi = theano.shared(uni(-np.sqrt(1.0/(nin*nhid)), np.sqrt(1.0/(nin*nhid)),(4, nin, nhid)).astype(theano.config.floatX),name='Wi')
    self.Wh = theano.shared(uni(-np.sqrt(1.0/(nhid**2)), np.sqrt(1.0/(nhid**2)),(4, nhid, nhid)).astype(theano.config.floatX),name='Wh')
    self.Wc = theano.shared(diag_constructor([-np.sqrt(1.0/(nhid**2)), np.sqrt(1.0/(nhid**2))],nhid,3),name='Wc')
    self.b = theano.shared(np.zeros((4,nhid)), name='b')

    self.Wy = theano.shared(uni(-np.sqrt(1.0/(nhid*nout)), np.sqrt(1.0/(nhid*nout)),(nhid,nout)).astype(theano.config.floatX),name='Wy')
    self.by = theano.shared(np.zeros(nout), name='by')

    self.params = [self.Wi, self.Wh, self.Wc, self.b, self.Wy, self.by]

    def recurrent_step(x_t,b_tm1,s_tm1):
      """
      Define the recurrent step.
      args:
          - x_t: the current sequence
          - b_tm1: the previous b_t (b_{t minus 1})
          - s_tml: the previous s_t (s_{t minus 1}) this is the state of the cell
      """
      # Input 
      b_L = T.nnet.sigmoid(T.dot(x_t, self.Wi[0]) + T.dot(b_tm1,self.Wh[0]) + T.dot(s_tm1, self.Wc[0]) + self.b[0])
      # Forget
      b_Phi = T.nnet.sigmoid(T.dot(x_t,self.Wi[1]) + T.dot(b_tm1,self.Wh[1]) + T.dot(s_tm1, self.Wc[1]) + self.b[1])
      # Cell 
      a_Cell = T.dot(x_t, self.Wi[2]) + T.dot(b_tm1, self.Wh[2]) + self.b[2]
      s_t = b_Phi * s_tm1 + b_L*T.tanh(a_Cell)
      # Output 
      b_Om = T.nnet.sigmoid(T.dot(x_t, self.Wi[3]) + T.dot(b_tm1,self.Wh[3]) + T.dot(s_t, self.Wc[2]) + self.b[3])
      # Final output (What gets sent to the next step in the recurrence) 
      b_Cell = b_Om*T.tanh(s_t)
      # Sequence output
      o_t = T.nnet.softmax(T.dot(b_Cell, self.Wy) + self.by)

      return b_Cell, s_t, o_t 

    out, _ = theano.scan(recurrent_step,
              truncate_gradient=truncate,
              sequences = X,
              outputs_info=[
                              {'initial':T.zeros((X.shape[1],nhid))},
                              {'initial':T.zeros((X.shape[1],nhid))},
                              {'initial':None}
                          ],
              n_steps=X.shape[0])

    self.b_out = out[0]
    self.pred = out[2]
