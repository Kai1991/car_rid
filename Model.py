from keras.layers import Dense,Input,concatenate,Lambda
from keras.models import Model,load_model
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from loss import identity_loss,triplet_loss
from  keras.applications import ResNet50
from data_gen import gen_data_train_wrap,gen_data_val_wrap


class CAR_RID_MODEL(object):
    def __init__(self,config,mode):
        self.config = config
        self.mode = mode
        self.model = self.build_model()

    def build_model(self):
        back_bone = ResNet50(include_top=False, weights= "imagenet",
               input_tensor=None, input_shape=(self.config.IMG_WIDTH, self.config.IMG_HEIGHT, 3),pooling=None)
        f_base = back_bone.get_layer(index = -1).output

        f_acs = Dense(1024, name='f_acs')(f_base)
        f_model = Dense(self.config.NBR_MODELS, activation='softmax', name='predictions_model')(f_acs)
        f_color = Dense(self.config.NBR_COLORS, activation='softmax', name='predictions_color')(f_acs)

        if(self.mode == "brand_color"):
            return Model(outputs = [f_model, f_color], inputs = back_bone.input)
        elif(self.mode == "branch_color_trip"):
            # 输入
            anchor = back_bone.input
            positive = Input(shape=(self.config.IMG_WIDTH, self.config.IMG_HEIGHT, 3), name='positive')
            negative = Input(shape=(self.config.IMG_WIDTH, self.config.IMG_HEIGHT, 3), name='negative')

            f_sls1 = Dense(1024,name='sls1')(f_base)
            f_sls2 = concatenate([f_sls1,f_acs],axis=-1)
            f_sls2 = Dense(1024, name = 'sls2')(f_sls2)
            f_sls3 = Dense(256, name = 'sls3')(f_sls2)
            sls_branch = Model(inputs = back_bone.input, outputs = f_sls3)
            #输出
            f_sls3_anchor = sls_branch(anchor)
            f_sls3_positive = sls_branch(positive)
            f_sls3_negative = sls_branch(negative)

            loss_triplet = Lambda(triplet_loss,output_shape=(1,))([f_sls3_anchor,f_sls3_positive,f_sls3_negative])

            return Model(inputs=[anchor,positive,negative],outputs=[f_model,f_color,loss_triplet])

    def compile(self):
        optimizer = SGD(lr = self.config.LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)
        if(self.mode == "brand_color"):
            self.model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                  #loss_weights = [0.6, 0.4],
                  optimizer=optimizer, metrics=["accuracy"])
        elif(self.mode == "branch_color_triplet"):
            self.model.compile(loss=["categorical_crossentropy", "categorical_crossentropy", identity_loss],
                  optimizer=optimizer, metrics=["accuracy"])

    def train(self):
        #callback
        checkpoint = ModelCheckpoint(self.config.model_file_saved, verbose = 1)
        reduce_lr = ReduceLROnPlateau(monitor='val_'+self.config.monitor_index, factor=0.5,patience=5, verbose=1, min_lr=0.00001)
        early_stop = EarlyStopping(monitor='val_'+self.config.monitor_index, patience=15, verbose=1)

        #训练数据集
        train_gen = gen_data_train_wrap(self.config.train_path,self.config,self.mode)
        val_gen = gen_data_val_wrap(self.config.val_path,self.config,self.mode)

        self.model.fit_generator(generator=train_gen,steps_per_epoch = self.config.steps_per_epoch, 
                        epochs = self.config.NBR_EPOCHS, verbose = 1,
                        validation_data = val_gen,validation_steps = self.config.validation_steps,
                        callbacks = [checkpoint,reduce_lr,early_stop], initial_epoch = self.config.INITIAL_EPOCH,
                        max_queue_size = 100, workers = 10)