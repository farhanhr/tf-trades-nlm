from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def create_model_vgg16():
        
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  

    top_model = base_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(128, activation='relu')(top_model)
    top_model = Dropout(0.5)(top_model)
    outputs = Dense(4, activation='softmax')(top_model) 

    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model