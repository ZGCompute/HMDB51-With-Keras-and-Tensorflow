import os                                                                                                                                                                                                   
import sys                                                                                                                                                          
import functools

from keras.models import Sequential                                                                                                                                                                         
from keras.layers.wrappers import TimeDistributed                                                                                                                   
from keras.preprocessing import sequence                                                                                                                            
from keras.preprocessing.image import load_img, img_to_array                                                                                                        
from keras.preprocessing.image import ImageDataGenerator                                                                                                            
from keras.layers.core import Dense, Dropout, Flatten, Activation                                                                                                   
from keras.layers.convolutional import Conv2D, MaxPooling2D                                                                                                         
from keras.layers.recurrent import LSTM                                                                                                                             
from keras.layers.pooling import GlobalAveragePooling1D

from keras.layers import BatchNormalization, Lambda, GlobalAveragePooling1D, Average                      
from keras.optimizers import SGD, Adam                                                                                                                              
from keras import layers, models, applications                                                                                                                      
from keras.utils.training_utils import multi_gpu_model
                                                                                                                                                                                                                                                                                                                
import keras                                                                                                                                                        
from keras.callbacks import ModelCheckpoint                                                                                                                         
from keras.callbacks import EarlyStopping                                                                                                                           
from keras.preprocessing.image import ImageDataGenerator                                                                                                            
from keras.callbacks import TensorBoard 
                                                                                                                            
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4'                                                                                                                      
from time import time                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                                                               
from keras.utils import plot_model                                                                                                                                  
                                                                                                                                                                        
# add Ouva utils to path                                                                                                                                            
sys.path.append('/home/ubuntu/Ouva_utils')                                                                                                                          
from Ouva_Ml import * 
from PIL import Image                                                                                                                                              

# Run training with true Batch-norm mean                                                                                                                            
K.set_learning_phase(1)

class ActRec_CNN_LSTM():

    def __init__(self):
                                                                                                                                                                     
        # Set hyperparams                                                                                                                                                  
        self.num_gpus = 4                                                                                                                                                        
        self.batch_size = 40
        self.learning_rate=0.001
                                                                                                                                                                                       
        self.my_momentum=0.09                                                                                                                                                    
        self.num_epochs = 1000                                                                                                                                                   
        self.num_examples = 3300
        self.num_classes=51                                                                                                                                                 
        self.epoch_steps = (num_examples / batch_size) + 1   
        self.frames=12                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        self.base_path = '/datag/jpegs_256';                                                                                                                                     
        self.flow_path = '/datag/tvl1_flow';                                                                                                                                     

        self.video1 = layers.Input(shape=(None, 224,224,3),batch_shape=(batch_size, None, 224, 224, 3), name='video_input1') 
        self.video2 = layers.Input(shape=(None, 224,224,3),batch_shape=(batch_size, None, 224, 224, 3), name='video_input2')

        self.Y_classes = np.loadtxt('/home/ubuntu/Ouva_utils/train/hmdb51_int_classes.txt')
        self.train_split_file = '/home/ubuntu/Ouva_utils/train/hmdb51_train_split.txt'
        self.test_split_file = '/home/ubuntu/Ouva_utils/train/hmdb51_test_split.txt' 

        self.tensorboard = TensorBoard(log_dir="/home/ubuntu/Ouva_utils/logs/{}".format(time()))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    # hack to use callbacks with multi_gpu                                                                                                                              
    class MyCbk(keras.callbacks.Callback):                                                                                                                              
                                                                                                                                                                        
        def __init__(self, model):                                                                                                                                      
            self.model_to_save = model                                                                                                                                  
                                                                                                                                                                        
        def on_epoch_end(self, epoch, logs=None):                                                                                                                       
            self.model_to_save.save('model_at_epoch_%d.h5' % epoch)                                                                                                     
                                                                                                                                                                        
        def chkPointer( self ):                                                                                                                                         
            ouput_file = '_lrcn_lr0001_mtm09_normx'                                                                                                             
            self.checkpointer = ModelCheckpoint(filepath='bestmodel' + output_file + ".hdf5", verbose=1, save_best_only=True)                                           
                                                                                                                                                                        
        def earlyStopeer( self ):                                                                                                                                       
            self.earlyStopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)                                                                               
                                                                                                                                                                        
    # Define custom video  onehot label encoder
    def load_Y_vid_lbl_onehot( fname ):                                                                                                                                 
                                                                                                                                                                        
        '''Load Y_labels.txt into N x NUM_CLASSES onehot matrix'''                                                                                                      
                                                                                                                                                                        
        Y = np.loadtxt(fname)                                                                                                                                           
                                                                                                                                                                        
        onehot_encoded = np.zeros([51])                                                                                                                                 
        onehot_encoded[int(Y[()])] = 1;                                                                                                                                 
                                                                                                                                                                        
        return onehot_encoded                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    # Load in hmdb51 train splits list                                                                                                                                  
    with open('/home/ubuntu/Ouva_utils/train/hmdb51_train_split.txt') as f:                                                                                             
        train_list = f.readlines()                                                                                                                                      
        train_list = [ i.strip() for i in train_list]                                                                                                                   
        train_list = [i[:-2] for i in train_list]                                                                                                                       
        train_list = [ i[:-4] for i in train_list]                                                                                                                      
                                                                                                                                                                        
    def load_Y_vid_lbl_onehot( fname ):                                                                                                                                                                         
                                                                                                                                                                        
        '''Load Y_labels.txt into N x NUM_CLASSES onehot matrix'''                                                                                                      
                                                                                                                                                                        
        Y = np.loadtxt(fname)                                                                                                                                           
                                                                                                                                                                        
        onehot_encoded = np.zeros([51])                                                                                                                                 
        onehot_encoded[int(Y[()])] = 1;                                                                                                                                 
        #onehot_encoded = np.expand_dims(onehot_encoded, axis=0)                                                                                                        
                                                                                                                                                                        
        return onehot_encoded 

    def build_rgb_model(self):                                                                                                                                       
        #model=Sequential()                                                                                                                                             
                                                                                                                                                                        
        x = TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(frames, 224, 224, 3))(self.video1)                                                              
        x = TimeDistributed(Activation('relu'))(x)                                                                                                                      
        x = TimeDistributed(Conv2D(32, (3, 3)))(x)                                                                                                                      
        x = TimeDistributed(Activation('relu'))(x)                                                                                                                      
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)                                                                                                          
        x = TimeDistributed(Dropout(0.25))(x)                                                                                                                           
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)                                                                                                          
        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dense(256))(x)                                                                                                                               
                                                                                                                             
        x = TimeDistributed(Dense(35, name="first_dense_rgb1" ))(x)                                                                                                     
                                                                                                                                                                        
        x = LSTM(20, return_sequences=True, name="lstm_layer_rgb1")(x)                                                                                                  
                                                                                                                                                                        
        x = TimeDistributed(Dense(num_classes), name="time_distr_dense_one_rgb1")(x)                                                                                    
        x = GlobalAveragePooling1D(name="global_avg_rgb1")(x)                                                                                                           
                                                                                                                                                                        
        model = Model(self.video1, x, name='rgb_lrcn1')                                                                                                                      
                                                                                                                                                                        
        return model

    def build_flow_model(self):                                                                                                                                       
        #model=Sequential()                                                                                                                                             
                                                                                                                                                                        
        x = TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(frames, 224, 224, 3))(self.video2)                                                              
        x = TimeDistributed(Activation('relu'))(x)                                                                                                                      
        x = TimeDistributed(Conv2D(32, (3, 3)))(x)                                                                                                                      
        x = TimeDistributed(Activation('relu'))(x)                                                                                                                      
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)                                                                                                          
        x = TimeDistributed(Dropout(0.25))(x)                                                                                                                           
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)                                                                                                          
        x = TimeDistributed(Flatten())(x)                                                                                                                               
        x = TimeDistributed(Dense(256))(x) 
                                                                                                                             
        x = TimeDistributed(Dense(35, name="first_dense_flow2" ))(x)                                                                                                     
                                                                                                                                                                        
        x = LSTM(20, return_sequences=True, name="lstm_layer_flow2")(x)                                                                                                  
                                                                                                                                                                        
        x = TimeDistributed(Dense(num_classes), name="time_distr_dense_one_flow2")(x)                                                                                    
        x = GlobalAveragePooling1D(name="global_avg_flow2")(x)                                                                                                           
                                                                                                                                                                        
        model = Model(self.video2, x, name='flow_lrcn2')                                                                                                                      
                                                                                                                                                                        
        return model                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                
    def build_shallow_model(self):                                                                                                                            
                                                                                                                                                                                                                                                                                                                            
        rgb_model = build_rgb_model(self.video1)                                                                                                                             
        flow_model = build_flow_model(self.video2)                                                                                                                           
                                                                                                                                                                        
        y = Average()([rgb_model(self.video1), flow_model(self.video2)])                                                                                                          
        model = Model(inputs=[self.video1, self.video2], outputs=y, name='ensemble')                                                                                              
                                                                                                                                                                        
        top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)                                                                                     
        top5_acc.__name__ = 'top5_acc'                                                                                                                                  
                                                                                                                                                                        
        model = multi_gpu_model(model, gpus=self.num_gpus)                                                                                                                   
        model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy',                                                              
                      metrics=['mae','accuracy', top5_acc])                                                                                                             
                                                                                                                                                                        
        cbk = self.MyCbk(model)                                                                                                                                              
        print(model.summary())                                                                                                                                          
                                                                                                                                                                        
        return model

    def get_deep_rgb_model(self):                                                                                                                                                                           
        # Define CNN-LSTM architecture                                                                                                                                                                
                                                                                                                                                                                                            
        # contruct the model                                                                                                                                                                          
        cnn_base_model = InceptionV3(weights='imagenet', include_top=False)                                                                                                                           
                                                                                                                                                             
        cnn_base_model.trainable=False                                                                                                                                                                
        encoded_frame = layers.TimeDistributed(cnn_base_model)(self.video1)                                                                                                                                
        encoded_frame = TimeDistributed(GlobalAveragePooling2D(input_shape=(None, 224,224,3)))(encoded_frame)                                                                                         
                                                                                        
        encoded_frame = TimeDistributed(Dense(256, activation='relu'))(encoded_frame)                                                                                                                 
        encoded_frame = layers.TimeDistributed(Flatten())(encoded_frame)                                                                                                                              
        encoded_frame = TimeDistributed(Dropout(0.25))(encoded_frame)                                                                                                                                 
        encoded_vid = LSTM(256, return_sequences=True, dropout=0.25)(encoded_frame)                                                                                                                   
                                                                                                                                 
        out = layers.TimeDistributed(Dense(1024, activation='relu'))(encoded_frame)                                                                                                                   
        out = layers.TimeDistributed(Dropout(0.5))(out)                                                                                                                                               
        out = layers.TimeDistributed(Dense(51, activation='softmax'))(out)                                                                                                                            
        out = GlobalAveragePooling1D()(out)                                                                                                                                                           
                                                                                                                                                                                                            
        model = Model(inputs=self.video1,outputs=out)                                                                                                                                                      
                                                                                                                                                                                                            
        return model

    def get_deep_flow_model(self):                                                                                                                                                                           
        # Define CNN-LSTM architecture                                                                                                                                                                
                                                                                                                                                                                                            
        # contruct the model                                                                                                                                                                          
        cnn_base_model = InceptionV3(weights='imagenet', include_top=False)                                                                                                                           
                                                                                                                                                             
        cnn_base_model.trainable=False                                                                                                                                                                
        encoded_frame = layers.TimeDistributed(cnn_base_model)(video1)                                                                                                                                
        encoded_frame = TimeDistributed(GlobalAveragePooling2D(input_shape=(None, 224,224,3)))(encoded_frame)                                                                                         
                                                                                        
        encoded_frame = TimeDistributed(Dense(256, activation='relu'))(encoded_frame)                                                                                                                 
        encoded_frame = layers.TimeDistributed(Flatten())(encoded_frame)                                                                          -                                                   
        encoded_frame = TimeDistributed(Dropout(0.25))(encoded_frame)                                                                                                                                 
        encoded_vid = LSTM(256, return_sequences=True, dropout=0.25)(encoded_frame)                                                                                                                   
                                                                                                                                 
        out = layers.TimeDistributed(Dense(1024, activation='relu'))(encoded_frame)                                                                                                                   
        out = layers.TimeDistributed(Dropout(0.5))(out)                                                                                                                                               
        out = layers.TimeDistributed(Dense(51, activation='softmax'))(out)                                                                                                                            
        out = GlobalAveragePooling1D()(out)                                                                                                                                                           
                                                                                                                                                                                                            
        model = Model(inputs=video2,outputs=out)                                                                                                                                                      
                                                                                                                                                                                                            
        return model


    def batch_iter(self):      
                                                                                                                                              
        split_data = np.genfromtxt(self.split_file, dtype='U', delimiter=" ")                                                                                                
        split_data = [i[0].replace('.avi', '') for i in split_data]                                                                                                     
        total_seq_num = len(split_data)                                                                                                                                 
        batch_size=40                                                                                                                                                   
        num_batches_per_epoch = int((total_seq_num - 1) / batch_size) + 1                                                                                               
                                                                                                                                                                        
        def data_generator():                                                                                                                                           
            while True:                                                                                                                                                 
                indices = np.random.permutation(np.arange(total_seq_num))                                                                                               
                                                                                                                                                                        
                while True: # for each batch                                                                                                                            
                    batch_num=1                                                      
                    start_index = batch_num * batch_size                                                                                                                
                    end_index = min((batch_num + 1) * batch_size, total_seq_num)                                                                                        
                                                                                                                                                                        
                    RGB = []                                                                                                                                            
                    FLOW = []                                                                                                                                           
                    Y = []                                                                                                                                              
                    for i in range(start_index, end_index): # for each sequence                                                                                         
                        image_dir = self.rgb_path + '/' + split_data[indices[i]]                                                                                             
                        flow_u_image_dir = self.flow_path + '/u/' + split_data[indices[i]]                                                                                   
                        flow_v_image_dir = self.flow_path + '/v/' + split_data[indices[i]]                                                                                   
                                                                                                                                                                        
                        listd = os.listdir(image_dir)                                                                                                                   
                        listd.sort(key=natural_keys)                                                                                                                    
                        listd.remove('X_train.h5')                                                                                                                      
                        listd.remove('Y_labels.txt')                                                                                                                    
                        seq_len = len(listd)                                                                                                                            
                                                                                                                                                                        
                        # load y example label                                                                                                                          
                        yfile = (image_dir + '/Y_labels.txt');                                                                                                          
                        Y_tmp = self.load_Y_vid_lbl_onehot(yfile)                                                                                                            
                        y = np.array([], dtype=np.float64).reshape(0, 51);                                                                                              
                                                                                                                                                                        
                        # To reduce the computational time, data augmentation is performed for each frame                                                               
                        reg_rgb = np.array([], dtype=np.float64).reshape(0, 224,224, 3);                                                                                
                        augs_rgb = np.array([], dtype=np.float64).reshape(0, 224,224, 3);                                                                               
                        augs_flow = np.array([], dtype=np.float64).reshape(0, 224,224, 3);
                        reg_flow = np.array([], dtype=np.float64).reshape(0, 224, 224, 3);                                                                              
                        for j in range(frames): # for each frame  
                                                                                                      
                            # Get frames at regular interval. start from frame index 1                                                                                  
                            frame = int(seq_len / frames * j) + 1                                                                                                       
                                                                                                                                                                        
                            # rgb image                                                                                                                                 
                            rgb_i = load_img(("%s/frame" + (6-len(str((int(frame)))))*"0" + str((int(frame))) + ".jpg") % (image_dir), target_size=(224, 224))          
                            rgb = img_to_array(rgb_i)                                                                                                                   
                            rgb = np.expand_dims(rgb, axis=0)                                                                                                           
                            reg_rgb = np.vstack([reg_rgb, rgb])                                                                                                         
                            rgb_flip_i = rgb_i.transpose(Image.FLIP_LEFT_RIGHT) # augmentation                                                                          
                            rgb_flip = img_to_array(rgb_flip_i)                                                                                                         
                            rgb_flip = np.expand_dims(rgb_flip, axis=0)                                                                                                 
                            augs_rgb = np.vstack([augs_rgb, rgb_flip])                                                                                                  
                            y = load_Y_vid_lbl_onehot('%s/Y_labels.txt' %(image_dir))                                                                                   
                                                                                                                                                                        
                            # flow image                                                                                                                                
                            flow_x_i = load_img(("%s/frame" + (6-len(str((int(frame)))))*"0" + str((int(frame))) + ".jpg") % (flow_u_image_dir), target_size=(224, 224))
                            flow_y_i = load_img(("%s/frame" + (6-len(str((int(frame)))))*"0" + str((int(frame))) + ".jpg") % (flow_v_image_dir), target_size=(224,224))
                            flow_x = img_to_array(flow_x_i)                                         
                            flow_x = flow_x[:,:,0]                                                  
                            flow_y = img_to_array(flow_y_i)                                         
                            flow_y = flow_y[:,:,0]                                                  
                                                                                                    
                            flow_x_flip_i = flow_x_i.transpose(Image.FLIP_LEFT_RIGHT) # augmentation
                            flow_y_flip_i = flow_y_i.transpose(Image.FLIP_LEFT_RIGHT) # augmentation
                            flow_x_flip = img_to_array(flow_x_flip_i)                            
                            flow_x_flip= flow_x_flip[:,:,0]                                 
                            flow_y_flip = img_to_array(flow_y_flip_i)
                            flow_y_flip = flow_y_flip[:,:,0]                       
    
                            # create third channel from magnitude of x and y
                            mag = np.sqrt(np.power(flow_x,2) + np.power(flow_y,2))
                            flow = np.stack([flow_x, flow_y, mag])
                            flow = flow.reshape(224,224,3)
                            flow = np.expand_dims(flow, axis=0)
                            reg_flow = np.vstack([reg_flow, flow])
                            mag_flip = np.sqrt(np.power(flow_x_flip,2) + np.power(flow_y_flip,2))
                            flow_flip = np.stack([flow_x_flip, flow_y_flip, mag_flip])
                            flow_flip = flow_flip.reshape(224,224,3)
                            flow_flip = np.expand_dims(flow_flip, axis=0)
                            augs_flow = np.vstack([augs_flow, flow_flip])
    
                        augs_rgb = augs_rgb / 255.
                        augs_flow = augs_flow / 255.                            
                        reg_rgb = reg_rgb / 255.
                        RGB.append(augs_rgb)
                        RGB.append(reg_rgb)
                        reg_flow = reg_flow / 255.
                        FLOW.append(reg_flow)                                                                                                          
                        FLOW.append(augs_flow) 
                        Y.append(y)
                        Y.append(y)
    
                    # Center data to mean and std of batch
                    mean = np.mean(RGB, axis=0)                                         
                    std = np.std(RGB,axis=0)                                            
                    RGB = [((i-mean) / std) for i in RGB]    
                    RGB = np.array(RGB)

                    mean_flow = np.mean(FLOW, axis=0)
                    std_flow = np.std(FLOW, axis=0)
                    FLOW = [((i-mean_flow) / std_flow) for i in FLOW]
                    FLOW = np.array(FLOW)    
                 
                    Y = np.array(Y)
                    
                    yield (RGB, Y)
                    batch_num +=1
                    
                    # Generate infinite data
                    if (batch_num == len(split_data)):
                        batch_num = 0
    
        return num_batches_per_epoch, data_generator()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                                                                                   
    def train(self): 

        # move to directory with training tars unpacked                                                                                                                     
        os.chdir(self.base_path);

        # Stack together training data                                                                                                                                      
        dirs = os.listdir('.');

        print("Examples in train list: %d" %(len(dirs)))

        # define top K accurcaey 
        top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)                                                                                                                                 
        top5_acc.__name__ = 'top5_acc'

        train_steps, train_batches = self.batch_iter(self.train_split_file, self.rgb_path, self.flow_path, self.frames)                                                                                    
        valid_steps, valid_batches = self.batch_iter(self.test_split_file, self.rgb_path, self.flow_path, self.frames) 

        # Build and compile the model
        model = build_rgb_model(self.video1)                                                                                                                                     
        model = multi_gpu_model(model, gpus=self.num_gpus)    
        cbk = self.MyCbk(model)                                                                                                                  
        model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipvalue=100), loss='categorical_crossentropy',                                                                 
                                                                                    metrics=['mae','accuracy', top5_acc])

        history_callback = model.fit_generator(train_batches, validation_data=valid_batches, use_multiprocessing=False, 
                                                                workers=1, shuffle=True, epochs=4,steps_per_epoch=50*2, 
                                                   validation_steps=50*2, verbose=1, callbacks=[cbk, self.tensorboard])

        # save updates after each epoch to .txt
        loss_history = history_callback.history["loss"]
        acc_history = history_callback.history["acc"]

        val_loss_history = history_callback.history["val_loss"]
        val_acc_history = history_callback.history["val_acc"]
        val_mae_history = history_callback.history["val_mae"]

        np_loss_history = np.array(loss_history)
        np_acc_history = np.array(acc_history)

        np_val_loss_history = np.array(val_loss_history)
        np_val_acc_history = np.array(val_acc_history)

        np.savetxt("loss_history.txt", np_loss_history, delimiter=",")
        np.savetxt("acc_history.txt", np_acc_history, delimiter=",")

        np.savetxt("val_loss_history.txt", np_val_loss_history, delimiter=",")
        np.savetxt("val_acc_history.txt", np_val_acc_history, delimiter=",")


if __name__ == '__main__':

    CNN_LSTM = ActRec_CNN_LSTM();
    CNN_LSTM.train();
  




    
