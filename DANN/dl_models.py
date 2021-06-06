from keras.layers import Input, Dense, Activation, BatchNormalization, PReLU, Dropout
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf

tf.random.set_seed(2)


def build_models(emb_dim, input_dim=2, label_num=65, dropout_rate=0.5, cm_lr=0.001, scm_lr=0.001, dsm_lr=0.001,
                 em_lr=0.001):
    """Creates three different models, one used for source only training, two used for domain adaptation"""
    inputs = Input(shape=(input_dim,))
    x4 = Dense(emb_dim, activation='linear')(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation("elu")(x4)

    source_classifier = Dense(label_num, activation='softmax', name="mo")(x4)
    domain_classifier = Dense(64, activation='linear', name="do4")(x4)     # 128
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Activation("elu", name="do6")(domain_classifier)
    domain_classifier = Dropout(dropout_rate)(domain_classifier)

    domain_classifier = Dense(2, activation='softmax', name="do")(domain_classifier)  # classify data domain

    comb_model = Model(inputs=inputs, outputs=[source_classifier, domain_classifier])
    comb_model.compile(optimizer=tf.optimizers.Adam(learning_rate=cm_lr),
                       loss={'mo': 'categorical_crossentropy', 'do': 'categorical_crossentropy'},
                       loss_weights={'mo': 1, 'do': 4}, metrics=['accuracy'])

    source_classification_model = Model(inputs=inputs, outputs=[source_classifier])
    source_classification_model.compile(optimizer=tf.optimizers.Adam(learning_rate=scm_lr),
                                        loss={'mo': 'categorical_crossentropy'}, metrics=['accuracy'], )

    domain_classification_model = Model(inputs=inputs, outputs=[domain_classifier])
    domain_classification_model.compile(optimizer=tf.optimizers.Adam(learning_rate=dsm_lr),
                                        loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])

    embeddings_model = Model(inputs=inputs, outputs=[x4])
    embeddings_model.compile(optimizer=tf.optimizers.Adam(learning_rate=em_lr), loss='categorical_crossentropy',
                             metrics=['accuracy'])

    return comb_model, source_classification_model, domain_classification_model, embeddings_model