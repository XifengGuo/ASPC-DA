from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
import numpy as np
from time import time
import math
from sklearn.cluster import KMeans
from MyModel import MyImageGenerator, autoencoder, generator
import metrics


class ASPC(object):
    def __init__(self, dims, n_clusters):
        self.dims = dims
        self.n_clusters = n_clusters
        self.centers = []
        self.y_pred = []
        self.datagen = MyImageGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)

        self.autoencoder, self.encoder = autoencoder(dims=dims)
        self.model = self.encoder
        self.pretrained = False

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp',
                 da_s1=False, verbose=1, use_multiprocessing=True):
        print('Pretraining......')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None and verbose > 0:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if epochs < 10 or epoch % int(epochs / 10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(index=int(len(self.model.layers) / 2)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        if not da_s1:
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        else:
            print('-=*' * 20)
            print('Using augmentation for pretraining')
            print('-=*' * 20)

            self.autoencoder.fit_generator(
                generator(self.datagen, x, batch_size=batch_size),
                steps_per_epoch=math.ceil(x.shape[0] / batch_size), epochs=epochs,
                callbacks=cb, verbose=verbose, use_multiprocessing=use_multiprocessing, workers=4)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def basic_clustering(self, x):
        """ Initialize a clustering result, i.e., labels and cluster centers.
        :param x: input data, shape=[n_samples, n_features]
        :return: labels and centers
        """
        print("Using k-means for initialization by default.")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, n_jobs=4)
        print(x.shape)
        y_pred = kmeans.fit_predict(X=x)
        centers = kmeans.cluster_centers_.astype(np.float32)
        return y_pred, centers

    def update_labels(self, x, centers):
        """ Update cluster labels.
        :param x: input data, shape=(n_samples, n_features)
        :param centers: cluster centers, shape=(n_cluster, n_features)
        :return: (labels, loss): labels indicate each sample belongs to which cluster. labels[i]=j means sample i
                 belongs to cluster j; loss, the average distance between samples and their responding centers
        """
        x_norm = np.reshape(np.sum(np.square(x), 1), [-1, 1])  # column vector
        center_norm = np.reshape(np.sum(np.square(centers), 1), [1, -1])  # row vector
        dists = x_norm - 2 * np.matmul(x, np.transpose(centers)) + center_norm  # |x-y|^2 = |x|^2 -2*x*y^T + |y|^2
        labels = np.argmin(dists, 1)
        losses = np.min(dists, 1)
        return labels, losses

    def compute_sample_weight(self, losses, t, T):
        lam = np.mean(losses) + t*np.std(losses) / T
        return np.where(losses < lam, 1., 0.)

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def predict(self, x, batch_size=256):
        return self.model.predict(x, batch_size)

    def predict_labels(self, x):  # predict cluster labels using the output of clustering layer
        return self.basic_clustering(self.predict(x))[0]

    def get_labels(self):
        return self.y_pred

    def compile(self, optimizer='sgd', loss='mse'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, batch_size=256, epochs=100,
            ae_weights=None, save_dir='result/temp', tol=0.001,
            use_sp=True, da_s2=False, use_multiprocessing=True):

        # prepare folder for saving results
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # pretraining
        t0 = time()
        if ae_weights is None and not self.pretrained:
            print('Pretraining AE...')
            self.pretrain(x, save_dir=save_dir)
            print('Pretraining time: %.1fs' % (time() - t0))
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully!')

        # initialization
        t1 = time()
        self.y_pred, self.centers = self.basic_clustering(self.predict(x))
        t2 = time()
        print('Time for initialization: %.1fs' % (t2 - t1))

        # logging file
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'acc', 'nmi', 'Ln', 'Lc'])
        logwriter.writeheader()

        net_loss = 0
        clustering_loss = 0
        time_train = 0
        sample_weight = np.ones(shape=x.shape[0])
        sample_weight[self.y_pred == -1] = 0  # do not use the noisy examples
        y_pred_last = np.copy(self.y_pred)
        result = None
        for epoch in range(epochs+1):
            """ Log and check stopping criterion """
            if y is not None:
                acc = np.round(metrics.acc(y, self.y_pred), 5)
                nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                print('Epoch-%d: ACC=%.4f, NMI=%.4f, Ln=%.4f, Lc=%.4f; time=%.1f' %
                      (epoch, acc, nmi, net_loss, clustering_loss, time_train))
                logwriter.writerow(dict(epoch=epoch, acc=acc, nmi=nmi, Ln=net_loss, Lc=clustering_loss))
                logfile.flush()

                # record the initial result
                if epoch == 0:
                    print('ASPC model saved to \'%s/model_init.h5\'' % save_dir)
                    self.model.save_weights(save_dir + '/model_init.h5')

                # check stop criterion
                delta_y = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if (epoch > 0 and delta_y < tol) or epoch >= epochs:
                    result = np.asarray([acc, nmi])
                    print('Training stopped: epoch=%d, delta_label=%.4f, tol=%.4f' % (epoch, delta_y, tol))
                    print('ASPC model saved to \'%s/model_final.h5\'' % save_dir)
                    print('-' * 30 + ' END: time=%.1fs ' % (time()-t0) + '-' * 30)
                    self.model.save_weights(save_dir + '/model_final.h5')
                    logfile.close()
                    break

            """ Step 1: train the network """
            t0_epoch = time()
            if da_s2:  # use data augmentation
                history = self.model.fit_generator(
                    generator(self.datagen, x, self.centers[self.y_pred], sample_weight, batch_size),
                    steps_per_epoch=math.ceil(x.shape[0] / batch_size), epochs=5 if np.any(self.y_pred == -1) and epoch==0 else 1,
                    use_multiprocessing=use_multiprocessing, workers=4, verbose=0)
            else:
                history = self.model.fit(x, y=self.centers[self.y_pred], batch_size=batch_size, epochs=1,
                                         sample_weight=sample_weight, verbose=0)
            net_loss = history.history['loss'][0]

            """ Step 2: update labels """
            self.y_pred, losses = self.update_labels(self.predict(x), self.centers)
            clustering_loss = np.mean(losses)

            """ Step 3: Compute sample weights """
            sample_weight = self.compute_sample_weight(losses, epoch, epochs) if use_sp else None

            time_train = time() - t0_epoch

        return result
