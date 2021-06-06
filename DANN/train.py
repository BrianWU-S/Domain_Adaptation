import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import tensorflow as tf
from dl_models import build_models
from utils import batch_generator, set_GPU_Memory_Limit, data_preprocessing
from tensorflow.keras.callbacks import LearningRateScheduler

tf.random.set_seed(2)


def train(Xs, ys, Xt, yt, enable_dann=True, n_iterations=15000, batch_size=64, dropout_rate=0.5, cm_lr=0.001,
          scm_lr=0.001, dsm_lr=0.001, em_lr=0.001, val_num=100):
    model, source_classification_model, domain_classification_model, embeddings_model = build_models(emb_dim=256,
                                                                                                     input_dim=
                                                                                                     np.shape(Xs)[1],
                                                                                                     label_num=len(
                                                                                                         np.unique(ys)),
                                                                                                     dropout_rate=dropout_rate,
                                                                                                     cm_lr=cm_lr,
                                                                                                     scm_lr=scm_lr,
                                                                                                     dsm_lr=dsm_lr,
                                                                                                     em_lr=em_lr)
    
    y_class_dummy = np.ones((len(Xt), 2))
    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))
    
    sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    sample_weights_adversarial = np.ones((batch_size * 2,))
    
    S_batches = batch_generator([Xs, to_categorical(ys)], batch_size)
    T_batches = batch_generator([Xt, np.zeros(shape=(len(Xt), 2))], batch_size)
    train_acc_list = []
    test_acc_list = []
    for i in range(n_iterations):
        # # print(y_class_dummy.shape, ys.shape)
        y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))
        
        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)
        
        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, np.zeros_like(y0)])
        
        adv_weights = []
        for layer in model.layers:
            if layer.name.startswith("do"):
                adv_weights.append(layer.get_weights())
        
        if enable_dann:
            # note - even though we save and append weights, the batchnorms moving means and variances
            # are not saved throught this mechanism
            stats = model.train_on_batch(X_adv, [y_class, y_adversarial_1],
                                         sample_weight=[sample_weights_class, sample_weights_adversarial])
            
            k = 0
            for layer in model.layers:
                if layer.name.startswith("do"):
                    layer.set_weights(adv_weights[k])
                    k += 1
            
            class_weights = []
            
            for layer in model.layers:
                if not layer.name.startswith("do"):
                    class_weights.append(layer.get_weights())
            
            stats2 = domain_classification_model.train_on_batch(X_adv, [y_adversarial_2])
            
            k = 0
            for layer in model.layers:
                if not layer.name.startswith("do"):
                    layer.set_weights(class_weights[k])
                    k += 1
        
        else:
            source_classification_model.train_on_batch(X0, y0)
        
        if (i + 1) % val_num == 0:
            y_test_hat_t = source_classification_model.predict(Xt).argmax(1)
            y_test_hat_s = source_classification_model.predict(Xs).argmax(1)
            print("Iteration %d, source accuracy =  %.3f, target accuracy = %.3f" % (
                i, accuracy_score(ys, y_test_hat_s), accuracy_score(yt, y_test_hat_t)))
            train_acc_list.append(accuracy_score(ys, y_test_hat_s))
            test_acc_list.append(accuracy_score(yt, y_test_hat_t))
            # if enable_dann:
            # print('stats:', stats)
    
    return embeddings_model, train_acc_list, test_acc_list


def finetune_model(src_features, src_labels, val_features, val_labels):
    # train test split
    X_train = src_features
    X_test = val_features
    Y_train = src_labels
    Y_test = val_labels
    val_num = 100
    print("X-train,X-test,Y-train,Y-test has shape:", np.shape(X_train), np.shape(X_test), np.shape(Y_train),
          np.shape(Y_test))
    # model training
    embedding_model_normal, train_acc_list, test_acc_list = train(Xs=X_train, ys=Y_train, Xt=X_test, yt=Y_test,
                                                                  enable_dann=False, n_iterations=5000,
                                                                  batch_size=64, dropout_rate=0.5, cm_lr=0.0005,
                                                                  scm_lr=0.0005, dsm_lr=0.0005, em_lr=0.0005,val_num=val_num)
    n_embedding_train = embedding_model_normal.predict(X_train)  # embedding model for visualization
    n_embedding_test = embedding_model_normal.predict(X_test)
    print("Train acc:", train_acc_list, '\n', "Test acc:", test_acc_list)
    print("Best acc:", np.max(test_acc_list), "Best epoch:", (np.argmax(test_acc_list) + 1) * val_num)


def domain_adaptation(src_features, src_labels, val_features, val_labels):
    # train test split
    X_train = src_features
    X_test = val_features
    Y_train = src_labels
    Y_test = val_labels
    val_num = 100
    print("X-train,X-test,Y-train,Y-test has shape:", np.shape(X_train), np.shape(X_test), np.shape(Y_train),
          np.shape(Y_test))
    embedding_model_transfer, train_acc_list, test_acc_list = train(Xs=X_train, ys=Y_train, Xt=X_test, yt=Y_test,
                                                                    enable_dann=True, n_iterations=5000, batch_size=32,
                                                                    dropout_rate=0.5, cm_lr=0.0001,
                                                                    scm_lr=0.0005, dsm_lr=0.0001, em_lr=0.0005,
                                                                    val_num=val_num)
    t_embedding_train = embedding_model_transfer.predict(X_train)
    t_embedding_test = embedding_model_transfer.predict(X_test)
    print("Train acc:", train_acc_list, '\n', "Test acc:", test_acc_list)
    print("Best acc:", np.max(test_acc_list), "Best epoch:", (np.argmax(test_acc_list) + 1) * val_num)


if __name__ == "__main__":
    set_GPU_Memory_Limit()
    # prepare data loader
    sd_features_AA, sd_features_CC, sd_features_PP, \
    td_features_AR, td_features_CR, td_features_PR, \
    sd_labels_AA, sd_labels_CC, sd_labels_PP, \
    td_labels_AR, td_labels_CR, td_labels_PR = data_preprocessing()
    print('Source features (AA):', np.shape(sd_features_AA), "Source labels:", np.shape(sd_labels_AA))
    print('Target features (AR):', np.shape(td_features_AR), "Target labels:", np.shape(td_labels_AR))
    # fine-tune model
    print("\n \n Fine tune Model: AA AR")
    finetune_model(src_features=sd_features_AA, src_labels=sd_labels_AA, val_features=td_features_AR,
                   val_labels=td_labels_AR)
    print("\n \n Fine tune Model: CC CR")
    finetune_model(src_features=sd_features_CC, src_labels=sd_labels_CC, val_features=td_features_CR,
                   val_labels=td_labels_CR)
    print("\n \n Fine tune Model: PP PR")
    finetune_model(src_features=sd_features_PP, src_labels=sd_labels_PP, val_features=td_features_PR,
                   val_labels=td_labels_PR)
    
    print("\n \n Domain Adaptation Model: AA AR")
    domain_adaptation(src_features=sd_features_AA, src_labels=sd_labels_AA, val_features=td_features_AR,
                      val_labels=td_labels_AR)
    print("\n \n Domain Adaptation Model: CC CR")
    domain_adaptation(src_features=sd_features_CC, src_labels=sd_labels_CC, val_features=td_features_CR,
                      val_labels=td_labels_CR)
    print("\n \n Domain Adaptation Model: PP PR")
    domain_adaptation(src_features=sd_features_PP, src_labels=sd_labels_PP, val_features=td_features_PR,
                      val_labels=td_labels_PR)

