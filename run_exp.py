from ASPC import ASPC
import os
import csv
from time import time
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from datasets import load_data


def run_exp(dbs, da_s1, da_s2, expdir, ae_weights_dir, trials=5, verbose=0,
            pretrain_epochs=50, finetune_epochs=50, use_multiprocessing=True):
    # Log files
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logfile = open(expdir + '/results.csv', 'a')
    logwriter = csv.DictWriter(logfile, fieldnames=['trials', 'acc', 'nmi', 'time'])
    logwriter.writeheader()

    # Begin training on different datasets
    for db in dbs:
        logwriter.writerow(dict(trials=db, acc='', nmi='', time=''))

        # load dataset
        x, y = load_data(db)

        # setting parameters
        n_clusters = len(np.unique(y))
        dims = [x.shape[-1], 500, 500, 2000, 10]

        # Training
        results = np.zeros(shape=[trials, 3], dtype=float)  # init metrics before finetuning
        for i in range(trials):  # base
            t0 = time()
            save_dir = os.path.join(expdir, db, 'trial%d' % i)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # prepare model
            model = ASPC(dims, n_clusters)
            model.compile(optimizer=Adam(0.0001), loss='mse')

            # pretraining
            ae_weights = 'ae_weights.h5'
            if ae_weights_dir is None:
                model.pretrain(x, y, optimizer=SGD(1.0, 0.9), epochs=pretrain_epochs,
                               save_dir=save_dir, da_s1=da_s1, verbose=verbose, use_multiprocessing=use_multiprocessing)
                ae_weights = os.path.join(save_dir, ae_weights)
            else:
                ae_weights = os.path.join(ae_weights_dir, db, 'trial%d' % i, ae_weights)

            # finetuning
            results[i, :2] = model.fit(x, y, epochs=finetune_epochs if db!='fmnist' else 10, 
                                       da_s2=da_s2, save_dir=save_dir, ae_weights=ae_weights,
                                       use_multiprocessing=use_multiprocessing)
            results[i, 2] = time() - t0

        for t, line in enumerate(results):
            logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], time=line[2]))
        mean = np.mean(results, 0)
        logwriter.writerow(dict(trials='avg', acc=mean[0], nmi=mean[1], time=mean[2]))
        logfile.flush()

    logfile.close()


if __name__=="__main__":
    # Global experiment settings
    expdir = 'result'
    ae_weight_root = None  # 'result'
    trials = 5
    verbose = 0
    dbs = ['mnist', 'mnist-test', 'usps', 'fmnist']  
    pretrain_epochs = 500
    finetune_epochs = 100
    use_multiprocessing = True  # if encounter errors, set it to False

    run_exp(dbs, da_s1=True, da_s2=True,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            use_multiprocessing=use_multiprocessing,
            expdir=expdir,
            ae_weights_dir=ae_weight_root,
            verbose=verbose, trials=trials)
