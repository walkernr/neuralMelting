# -*- coding: utf-8 -*-
"""
Created on Wed May 8 17:53:27 2019

@author: Nicholas
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelBinarizer
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
from sklearn.model_selection import train_test_split
from scipy.odr import ODR, Model as ODRModel, RealData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-pt', '--plot', help='plot results', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
    parser.add_argument('-g', '--gpu', help='gpu run (will default to cpu if unable)', action='store_true')
    parser.add_argument('-ad', '--anomaly_detection', help='anomaly detection for embedding', action='store_true')
    parser.add_argument('-nt', '--threads', help='number of threads',
                        type=int, default=20)
    parser.add_argument('-n', '--name', help='simulation name',
                            type=str, default='remcmc_init')
    parser.add_argument('-e', '--element', help='element choice',
                        type=str, default='LJ')
    parser.add_argument('-un', '--unsuper_samples', help='number of samples per phase point (manifold)',
                        type=int, default=512)
    parser.add_argument('-sn', '--super_samples', help='number of samples per phase point (variational autoencoder)',
                        type=int, default=1024)
    parser.add_argument('-sc', '--scaler', help='feature scaler',
                        type=str, default='global')
    parser.add_argument('-ld', '--latent_dimension', help='latent dimension of the variational autoencoder',
                        type=int, default=8)
    parser.add_argument('-mf', '--manifold', help='manifold learning method',
                        type=str, default='tsne')
    parser.add_argument('-cl', '--clustering', help='clustering method',
                        type=str, default='dbscan')
    parser.add_argument('-nc', '--clusters', help='number of clusters (neighbor criterion eps for dbscan)',
                        type=float, default=2e-3)
    parser.add_argument('-bk', '--backend', help='keras backend',
                        type=str, default='tensorflow')
    parser.add_argument('-opt', '--optimizer', help='optimization function',
                        type=str, default='nadam')
    parser.add_argument('-lss', '--loss', help='loss function',
                        type=str, default='mse')
    parser.add_argument('-ep', '--epochs', help='number of epochs',
                        type=int, default=1024)
    parser.add_argument('-lr', '--learning_rate', help='learning rate for neural network',
                        type=float, default=1e-4)
    parser.add_argument('-sd', '--random_seed', help='random seed for sample selection and learning',
                        type=int, default=256)
    args = parser.parse_args()
    return (args.verbose, args.plot, args.parallel, args.gpu, args.anomaly_detection, args.threads, args.name, args.element,
            args.unsuper_samples, args.super_samples, args.scaler, args.latent_dimension, args.manifold, args.clustering,
            args.clusters, args.backend, args.optimizer, args.loss, args.epochs, args.learning_rate, args.random_seed)


def write_specs():
    if VERBOSE:
        print(100*'-')
        print('input summary')
        print(100*'-')
        print('plot:                      %d' % PLOT)
        print('parallel:                  %d' % PARALLEL)
        print('gpu:                       %d' % GPU)
        print('anomaly detection:         %d' % AD)
        print('threads:                   %d' % THREADS)
        print('name:                      %s' % NAME)
        print('element:                   %s' % EL)
        print('random seed:               %d' % SEED)
        print('unsuper samples:           %d' % UNS)
        print('super samples:             %d' % SNS)
        print('scaler:                    %s' % SCLR)
        print('latent dimension:          %d' % LD)
        print('manifold learning:         %s' % MNFLD)
        print('clustering:                %s' % CLST)
        if CLST == 'dbscan':
            print('neighbor eps:              %.2e' % NC)
        else:
            print('clusters:                  %d' % NC)
        print('backend:                   %s' % BACKEND)
        print('network:                   %s' % 'cnn2d')
        print('optimizer:                 %s' % OPT)
        print('loss function:             %s' % LSS)
        print('epochs:                    %d' % EP)
        print('learning rate:             %.2e' % LR)
        print('fitting function:          %s' % 'logistic')
        print(100*'-')
    with open(OUTPREF+'.out', 'w') as out:
        out.write(100*'-' + '\n')
        out.write('input summary\n')
        out.write(100*'-' + '\n')
        out.write('plot:                      %d\n' % PLOT)
        out.write('parallel:                  %d\n' % PARALLEL)
        out.write('gpu:                       %d\n' % GPU)
        out.write('anomaly detection:         %d\n' % AD)
        out.write('threads:                   %d\n' % THREADS)
        out.write('name:                      %s\n' % NAME)
        out.write('element:                   %s\n' % EL)
        out.write('random seed:               %d\n' % SEED)
        out.write('unsuper samples:           %d\n' % UNS)
        out.write('super samples:             %d\n' % SNS)
        out.write('scaler:                    %s\n' % SCLR)
        out.write('latent dimension:          %d\n' % LD)
        out.write('manifold learning:         %s\n' % MNFLD)
        out.write('clustering:                %s\n' % CLST)
        if CLST == 'dbscan':
            out.write('neighbor eps:              %.2e\n' % NC)
        else:
            out.write('clusters:                  %d\n' % NC)
        out.write('backend:                   %s\n' % BACKEND)
        out.write('network:                   %s\n' % 'cnn2d')
        out.write('optimizer:                 %s\n' % OPT)
        out.write('loss function:             %s\n' % LSS)
        out.write('epochs:                    %d\n' % EP)
        out.write('learning rate:             %.2e\n' % LR)
        out.write('fitting function:          %s\n' % 'logistic')
        out.write(100*'-' + '\n')


def logistic(beta, t):
    ''' returns logistic sigmoid '''
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))


def odr_fit(func, dom, mrng, srng, pg):
    ''' performs orthogonal distance regression '''
    dat = RealData(dom, mrng, EPS*np.ones(len(dom)), srng+EPS)
    mod = ODRModel(func)
    odr = ODR(dat, mod, pg)
    odr.set_job(fit_type=0)
    fit = odr.run()
    popt = fit.beta
    perr = fit.sd_beta
    ndom = 128
    fdom = np.linspace(np.min(dom), np.max(dom), ndom)
    fval = func(popt, fdom)
    return popt, perr, fdom, fval


def gauss_sampling(beta):
    z_mean, z_log_var = beta
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def build_variational_autoencoder():
    if VERBOSE:
        print('building variational autoencoder network')
        print(100*'-')
    # encoder layers
    input = Input(shape=(NSP, NSP, NSP, 1), name='encoder_input')
    conv0 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(input)
    conv1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(2, 2, 2))(conv0)
    conv2 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(conv1)
    conv3 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu',
                   kernel_initializer='he_normal', padding='same', strides=(2, 2, 2))(conv2)
    shape = K.int_shape(conv3)
    fconv = Flatten()(conv3)
    d0 = Dense(8*LD, activation='relu')(fconv)
    z_mean = Dense(LD, name='z_mean')(d0)
    z_log_var = Dense(LD, name='z_log_std')(d0) # more numerically stable to use log(var_z)
    z = Lambda(gauss_sampling, output_shape=(LD,), name='z')([z_mean, z_log_var])
    # construct encoder
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    if VERBOSE:
        print('encoder network summary')
        print(100*'-')
        encoder.summary()
        print(100*'-')
    # decoder layers
    latent_input = Input(shape=(LD,), name='z_sampling')
    d1 = Dense(np.prod(shape[1:]), activation='relu')(latent_input)
    rd1 = Reshape(shape[1:])(d1)
    convt0 = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(rd1)
    convt1 = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(2, 2, 2))(convt0)
    convt2 = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(convt1)
    convt3 = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), activation='relu',
                             kernel_initializer='he_normal', padding='same', strides=(2, 2, 2))(convt2)
    output = Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), activation='tanh',
                             kernel_initializer='he_normal', padding='same', name='decoder_output')(convt3)
    # construct decoder
    decoder = Model(latent_input, output, name='decoder')
    if VERBOSE:
        print('decoder network summary')
        print(100*'-')
        decoder.summary()
        print(100*'-')
    # construct vae
    output = decoder(encoder(input)[2])
    vae = Model(input, output, name='vae_mlp')
    reconstruction_losses = {'bc': lambda a, b: binary_crossentropy(a, b),
                             'mse': lambda a, b: mse(a, b)}
    # vae loss
    reconstruction_loss = NSP**3*reconstruction_losses[LSS](K.flatten(input), K.flatten(output))
    kl_loss = -0.5*K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss+kl_loss)
    vae.add_loss(vae_loss)
    # compile vae
    vae.compile(optimizer=OPTS[OPT])
    # return vae networks
    return encoder, decoder, vae


def random_selection(dat, p, t, pe, ke, vol, nrho, ns):
    pn, tn, _, _, _, _ = dat.shape
    idat = np.zeros((pn, tn, ns), dtype=np.uint16)
    if VERBOSE:
        print('selecting random classification samples from full data')
        print(100*'-')
    for i in tqdm(range(pn), disable=not VERBOSE):
        for j in tqdm(range(tn), disable=not VERBOSE):
                idat[i, j] = np.random.permutation(dat[i, j].shape[0])[:ns]
    if VERBOSE:
        print('\n'+100*'-')
    sldat = np.array([[dat[i, j, idat[i, j], :, :, :] for j in range(tn)] for i in range(pn)])
    slp = np.array([[p[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slt = np.array([[t[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slpe = np.array([[pe[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slke = np.array([[ke[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slvol = np.array([[vol[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slnrho = np.array([[nrho[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    return sldat, slp, slt, slpe, slke, slvol, slnrho


def inlier_selection(dat, p, t, pe, ke, vol, nrho, ns):
    pn, tn, _, _, _ = dat.shape
    if AD:
        lof = LocalOutlierFactor(contamination='auto', n_jobs=THREADS)
    idat = np.zeros((pn, tn, ns), dtype=np.uint16)
    if VERBOSE:
        print('selecting inlier samples from classification data')
        print(100*'-')
    for i in tqdm(range(pn), disable=not VERBOSE):
        for j in tqdm(range(tn), disable=not VERBOSE):
                if AD:
                    fpred = lof.fit_predict(dat[i, j, :, 0])
                    try:
                        idat[i, j] = np.random.choice(np.where(fpred==1)[0], size=ns, replace=False)
                    except:
                        idat[i, j] = np.argsort(lof.negative_outlier_factor_)[:ns]
                else:
                    idat[i, j] = np.random.permutation(dat[i, j].shape[0])[:ns]
    if VERBOSE:
        print('\n'+100*'-')
    sldat = np.array([[dat[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slp = np.array([[p[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slt = np.array([[t[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slpe = np.array([[pe[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slke = np.array([[ke[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slvol = np.array([[vol[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    slnrho = np.array([[nrho[i, j, idat[i, j]] for j in range(tn)] for i in range(pn)])
    return sldat, slp, slt, slpe, slke, slvol, slnrho

if __name__ == '__main__':
    # parse command line arguments
    (VERBOSE, PLOT, PARALLEL, GPU, AD, THREADS, NAME, EL,
     UNS, SNS, SCLR, LD, MNFLD, CLST, NC,
     BACKEND, OPT, LSS, EP, LR, SEED) = parse_args()
    if CLST == 'dbscan':
        NCS = '%.0e' % NC
    else:
        NC = int(NC)
        NCS = '%d' % NC
    CWD = os.getcwd()
    EPS = 0.025
    # number of phases
    NPH = 2
    # number of embedding dimensions
    ED = 2

    np.random.seed(SEED)
    # environment variables
    os.environ['KERAS_BACKEND'] = BACKEND
    if BACKEND == 'tensorflow':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow import set_random_seed
        set_random_seed(SEED)
    if PARALLEL:
        if not GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['MKL_NUM_THREADS'] = str(THREADS)
        os.environ['GOTO_NUM_THREADS'] = str(THREADS)
        os.environ['OMP_NUM_THREADS'] = str(THREADS)
        os.environ['openmp'] = 'True'
    else:
        THREADS = 1
    if GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from keras.models import Model
    from keras.layers import Input, Lambda, Dense, Conv3D, Conv3DTranspose, Flatten, Reshape
    from keras.losses import binary_crossentropy, mse
    from keras.optimizers import SGD, Adadelta, Adam, Nadam
    from keras.callbacks import History, CSVLogger, ReduceLROnPlateau
    from keras import backend as K
    if PLOT:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
        plt.rc('font', family='sans-serif')
        FTSZ = 28
        FIGW = 16
        PPARAMS = {'figure.figsize': (FIGW, FIGW),
                   'lines.linewidth': 4.0,
                   'legend.fontsize': FTSZ,
                   'axes.labelsize': FTSZ,
                   'axes.titlesize': FTSZ,
                   'axes.linewidth': 2.0,
                   'xtick.labelsize': FTSZ,
                   'xtick.major.size': 20,
                   'xtick.major.width': 2.0,
                   'ytick.labelsize': FTSZ,
                   'ytick.major.size': 20,
                   'ytick.major.width': 2.0,
                   'font.size': FTSZ}
        plt.rcParams.update(PPARAMS)
        SCALE = lambda a, b: (a-np.min(b))/(np.max(b)-np.min(b))
        CM = plt.get_cmap('plasma')

    # lattice type
    LAT = {'Ti': 'bcc',
           'Al': 'fcc',
           'Ni': 'fcc',
           'Cu': 'fcc',
           'LJ': 'fcc'}

    PREF = CWD+'/%s.%s.%s.lammps' % (NAME, EL.lower(), LAT[EL])
    OUTPREF = PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%s.%s.%s.%04d' % \
              (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, MNFLD, CLST, NCS, SEED)
    write_specs()

    LDIR = os.listdir()

    try:
        CDAT = np.load(PREF+'.%04d.%d.cdf.c.npy' % (SNS, SEED))
        CP = np.load(PREF+'.%04d.%d.p.c.npy' % (SNS, SEED))
        CT = np.load(PREF+'.%04d.%d.t.c.npy' % (SNS, SEED))
        CPE = np.load(PREF+'.%04d.%d.pe.c.npy' % (SNS, SEED))
        CKE = np.load(PREF+'.%04d.%d.ke.c.npy' % (SNS, SEED))
        CVOL = np.load(PREF+'.%04d.%d.vol.c.npy' % (SNS, SEED))
        CNRHO = np.load(PREF+'.%04d.%d.nrho.c.npy' % (SNS, SEED))
        if VERBOSE:
            # print(100*'-')
            print('selected classification samples loaded from file')
            print(100*'-')
    except:
        DAT = np.load(PREF+'.cdf.npy')
        P = np.load(PREF+'.virial.npy')
        T = np.load(PREF+'.temp.npy')
        PE = np.load(PREF+'.pe.npy')
        KE = np.load(PREF+'.ke.npy')
        VOL = np.load(PREF+'.vol.npy')
        NRHO = np.load(PREF+'.nrho.npy')
        if VERBOSE:
            # print(100*'-')
            print('full dataset loaded from file')
            print(100*'-')
        CDAT, CP, CT, CPE, CKE, CVOL, CNRHO = random_selection(DAT, P, T, PE, KE, VOL, NRHO, SNS)
        del DAT, P, T, PE, KE, VOL, NRHO
        np.save(PREF+'.%04d.%d.cdf.c.npy' % (SNS, SEED), CDAT)
        np.save(PREF+'.%04d.%d.p.c.npy' % (SNS, SEED), CP)
        np.save(PREF+'.%04d.%d.t.c.npy' % (SNS, SEED), CT)
        np.save(PREF+'.%04d.%d.pe.c.npy' % (SNS, SEED), CPE)
        np.save(PREF+'.%04d.%d.ke.c.npy' % (SNS, SEED), CKE)
        np.save(PREF+'.%04d.%d.vol.c.npy' % (SNS, SEED), CVOL)
        np.save(PREF+'.%04d.%d.nrho.c.npy' % (SNS, SEED), CNRHO)
        if VERBOSE:
            print('selected classification samples generated')
            print(100*'-')
    TP = np.load(PREF+'.virial.trgt.npy')
    TT = np.load(PREF+'.temp.trgt.npy')
    NP, NT, _, NSP, _, _ = CDAT.shape
    NCH = 1
    SSHP0 = (NP, NT, SNS, NSP, NSP, NSP, NCH)
    SSHP1 = (NP*NT*SNS, NSP, NSP, NSP, NCH)
    SSHP2 = (NP*NT*SNS, NSP**3*NCH)
    SSHP3 = (NP, NT, SNS, 2, LD)
    SSHP4 = (NP*NT*SNS, 2, LD)

    # scaler dictionary
    SCLRS = {'minmax':MinMaxScaler(feature_range=(0, 1)),
             'standard':StandardScaler(),
             'robust':RobustScaler(),
             'tanh':TanhScaler()}

    try:
        SCDAT = np.load(PREF+'.%04d.%s.%04d.cdf.sc.npy' \
                        % (SNS, SCLR, SEED)).reshape(*SSHP1)
        if VERBOSE:
            print('scaled selected classification samples loaded from file')
            print(100*'-')
    except:
        if SCLR == 'none':
            SCDAT = CDAT.reshape(*SSHP1)
        if SCLR == 'global':
            SCDAT = ((CDAT-CDAT.min())/(CDAT.max()-CDAT.min())).reshape(*SSHP1)
        else:
            SCDAT = SCLRS[SCLR].fit_transform(CDAT.reshape(*SSHP2)).reshape(*SSHP1)
        np.save(PREF+'.%04d.%s.%04d.cdf.sc.npy' % (SNS, SCLR, SEED), SCDAT.reshape(*SSHP0))
        if VERBOSE:
            print('scaled selected classification samples computed')
            print(100*'-')

    OPTS = {'sgd': SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=True),
            'adadelta': Adadelta(lr=LR, rho=0.95, epsilon=None, decay=0.0),
            'adam': Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
            'nadam': Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)}

    ENC, DEC, VAE = build_variational_autoencoder()
    try:
        VAE.load_weights(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.vae.wt.h5' \
                         % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), by_name=True)
        TLOSS = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.vae.loss.trn.npy' \
                        % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        VLOSS = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.vae.loss.val.npy' \
                        % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        if VERBOSE:
            print('variational autoencoder trained weights loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('variational autoencoder training on scaled selected classification samples')
            print(100*'-')
        CSVLG = CSVLogger(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.vae.log.csv'
                          % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), append=True, separator=',')
        LR_DECAY = ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=VERBOSE)
        TRN, VAL = train_test_split(SCDAT, test_size=0.25, shuffle=True)
        VAE.fit(x=TRN, y=None, validation_data=(VAL, None), epochs=EP, batch_size=64,
                shuffle=True, verbose=VERBOSE, callbacks=[CSVLG, LR_DECAY, History()])
        del TRN, VAL
        TLOSS = VAE.history.history['loss']
        VLOSS = VAE.history.history['val_loss']
        VAE.save_weights(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.vae.wt.h5'
                         % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.vae.loss.trn.npy'
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), TLOSS)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.vae.loss.val.npy'
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), VLOSS)
        if VERBOSE:
            print(100*'-')
            print('variational autoencoder weights trained')
            print(100*'-')

    if VERBOSE:
        print('variational autoencoder training history information')
        print(100*'-')
        print('| epoch | training loss | validation loss |')
        print(100*'-')
        for i in range(EP):
            print('%02d %.2f %.2f' % (i, TLOSS[i], VLOSS[i]))
        print(100*'-')

    with open(OUTPREF+'.out', 'a') as out:
        out.write('variational autoencoder training history information\n')
        out.write(100*'-' + '\n')
        out.write('| epoch | training loss | validation loss |\n')
        out.write(100*'-' + '\n')
        for i in range(EP):
            out.write('%02d %.2f %.2f\n' % (i, TLOSS[i], VLOSS[i]))
        out.write(100*'-' + '\n')

    try:
        ZENC = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.npy'
                       % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED)).reshape(*SSHP4)
        ZDEC = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zdec.npy'
                       % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        # del SCDAT
        if VERBOSE:
            print('z encodings of scaled selected classification samples loaded from file')
            print(100*'-')
            print('error: %f' % ERR)
            print(100*'-')
    except:
        if VERBOSE:
            print('predicting z encodings of scaled selected classification samples')
            print(100*'-')
        ZENC = np.array(ENC.predict(SCDAT, verbose=VERBOSE))
        ZDEC = np.array(DEC.predict(ZENC[2, :, :], verbose=VERBOSE))
        ZENC = np.swapaxes(ZENC, 0, 1)[:, :2, :]
        ZENC[:, 1, :] = np.exp(0.5*ZENC[:, 1, :])
        # del SCDAT
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.npy'
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), ZENC.reshape(*SSHP3))
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zdec.npy'
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), ZDEC.reshape(*SSHP0))
    ERR = np.sqrt(np.square(ZDEC-SCDAT))
    MERR = np.mean(ERR)
    SERR = np.std(ERR)
    MXERR = np.max(ERR)
    MNERR = np.min(ERR)
    KLD = np.sum(1+np.log(np.square(ZENC[:, 1, :])-np.square(ZENC[:, 0, :])-np.square(ZENC[:, 1, :]), axis=-1)
    if VERBOSE:
        print(100*'-')
        print('z encodings of scaled selected classification samples predicted')
        print(100*'-')
        print('mean sig:   %f' % np.mean(SCDAT))
        print('stdv sig:   %f' % np.std(SCDAT))
        print('max sig     %f' % np.max(SCDAT))
        print('min sig     %f' % np.min(SCDAT))
        print('mean error: %f' % MERR)
        print('stdv error: %f' % SERR)
        print('max error:  %f' % MXERR)
        print('min error:  %f' % MNERR)
        print('kl div:     %f' % KLD)
        print(100*'-')
    with open(OUTPREF+'.out', 'a') as out:
        out.write('fitting errors\n')
        out.write(100*'-'+'\n')
        out.write('mean sig:   %f\n' % np.mean(SCDAT))
        out.write('stdv sig:   %f\n' % np.std(SCDAT))
        out.write('max sig     %f\n' % np.max(SCDAT))
        out.write('min sig     %f\n' % np.min(SCDAT))
        out.write('mean error: %f\n' % MERR)
        out.write('stdv error: %f\n' % SERR)
        out.write('max error:  %f\n' % MXERR)
        out.write('min error:  %f\n' % MNERR)
        out.write('kl div:     %f\n' % KLD)
        out.write(100*'-'+'\n')

    try:
        PZENC = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.pca.prj.npy'
                        % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED)).reshape(*SSHP4)
        CZENC = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.pca.cmp.npy'
                        % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        VZENC = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.pca.var.npy'
                        % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED))
        if VERBOSE:
            print('pca projections of z encodings  loaded from file')
            print(100*'-')
    except:
        if VERBOSE:
            print('pca projecting z encodings')
            print(100*'-')
        PCAZENC = PCA(n_components=LD)
        PZENC = np.zeros(SSHP4)
        CZENC = np.zeros((2, LD, LD))
        VZENC = np.zeros((2, LD))
        for i in range(2):
            PZENC[:, i, :] = PCAZENC.fit_transform(ZENC[:, i, :])
            CZENC[i, :, :] = PCAZENC.components_
            VZENC[i, :] = PCAZENC.explained_variance_ratio_
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.pca.prj.npy'
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), PZENC.reshape(*SSHP3))
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.pca.cmp.npy'
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), CZENC)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.zenc.pca.var.npy'
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED), VZENC)

    if VERBOSE:
        print('pca fit information')
        print(100*'-')
        for i in range(2):
            if i == 0:
                print('mean z fit')
            if i == 1:
                print('stdv z fit')
            print(100*'-')
            print('components')
            print(100*'-')
            for j in range(LD):
                print(LD*'%f ' % tuple(CZENC[i, j, :]))
            print(100*'-')
            print('explained variances')
            print(100*'-')
            print(LD*'%f ' % tuple(VZENC[i, :]))
            print(100*'-')
    with open(OUTPREF+'.out', 'a') as out:
        out.write('pca fit information\n')
        out.write(100*'-'+'\n')
        for i in range(2):
            if i == 0:
                out.write('mean z fit\n')
            if i == 1:
                out.write('stdv z fit\n')
            out.write(100*'-'+'\n')
            out.write('principal components\n')
            out.write(100*'-'+'\n')
            for j in range(LD):
                out.write(LD*'%f ' % tuple(CZENC[i, j, :]) + '\n')
            out.write(100*'-'+'\n')
            out.write('explained variances\n')
            out.write(100*'-'+'\n')
            out.write(LD*'%f ' % tuple(VZENC[i, :]) + '\n')
            out.write(100*'-'+'\n')

    def vae_plots():
        pref = PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d' % (SNS, SCLR, OPT, LSS, LD, EP, LR, SEED)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.scatter(ZENC[:, 0, 0], ZENC[:, 1, 0], c=CT.reshape(-1),
                   cmap=plt.get_cmap('plasma'), s=64, alpha=0.5, edgecolors='')
        plt.xlabel('VAE MU 0')
        plt.ylabel('VAE SIGMA 0')
        fig.savefig(pref+'.vae.prj.ld.png')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.scatter(PZENC[:, 0, 0], PZENC[:, 1, 0], c=CT.reshape(-1),
                   cmap=plt.get_cmap('plasma'), s=64, alpha=0.5, edgecolors='')
        plt.xlabel('PCA VAE MU 0')
        plt.ylabel('PCA SIGMA 0')
        fig.savefig(pref+'.vae.pca.prj.ld.png')
        plt.close()

        DIAGMLV = SCLRS['minmax'].fit_transform(np.mean(ZENC.reshape(NP, NT, SNS, 2*LD), 2).reshape(NP*NT, 2*LD)).reshape(NP, NT, 2, LD)
        DIAGSLV = SCLRS['minmax'].fit_transform(np.var(ZENC.reshape(NP, NT, SNS, 2*LD)/\
                                                CT[:, :, :, np.newaxis], 2).reshape(NP*NT, 2*LD)).reshape(NP, NT, 2, LD)

        DIAGMPLV = SCLRS['minmax'].fit_transform(np.mean(PZENC.reshape(NP, NT, SNS, 2*LD), 2).reshape(NP*NT, 2*LD)).reshape(NP, NT, 2, LD)
        for i in range(LD):
            if DIAGMPLV[0, 0, 0, i] > DIAGMPLV[-1, 0, 0, i]:
                DIAGMPLV[:, :, 0, i] = 1-DIAGMPLV[:, :, 0, i]
            if DIAGMPLV[int(NP/2), 0, 1, i] > DIAGMPLV[int(NP/2), -1, 1, i]:
                DIAGMPLV[:, :, 1, i] = 1-DIAGMPLV[:, :, 1, i]
        DIAGSPLV = SCLRS['minmax'].fit_transform(np.var(PZENC.reshape(NP, NT, SNS, 2*LD)/\
                                                 CT[:, :, :, np.newaxis], 2).reshape(NP*NT, 2*LD)).reshape(NP, NT, 2, LD)

        for i in range(2):
            for j in range(2):
                for k in range(LD):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        ax.imshow(DIAGMLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    if i == 1:
                        ax.imshow(DIAGSLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(TT.size), minor=True)
                    ax.set_yticks(np.arange(TP.size), minor=True)
                    plt.xticks(np.arange(TT.size), np.round(TT, 2), rotation=-60)
                    plt.yticks(np.arange(TP.size), np.round(TP, 2))
                    plt.xlabel('TEMP')
                    plt.ylabel('PRESS')
                    fig.savefig(pref+'.vae.diag.ld.%d.%d.%d.png' % (i, j, k))
                    plt.close()
        for i in range(2):
            for j in range(2):
                for k in range(LD):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    if i == 0:
                        ax.imshow(DIAGMPLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    if i == 1:
                        ax.imshow(DIAGSPLV[:, :, j, k], aspect='equal', interpolation='none', origin='lower', cmap=CM)
                    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
                    ax.set_xticks(np.arange(TT.size), minor=True)
                    ax.set_yticks(np.arange(TP.size), minor=True)
                    plt.xticks(np.arange(TT.size), np.round(TT, 2), rotation=-60)
                    plt.yticks(np.arange(TP.size), np.round(TP, 2))
                    plt.xlabel('TEMP')
                    plt.ylabel('PRESS')
                    fig.savefig(pref+'.vae.diag.ld.pca.%d.%d.%d.png' % (i, j, k))
                    plt.close()

    if PLOT:
        vae_plots()

    try:
        SLPZENC = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.zenc.pca.prj.inl.npy' \
                          % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED))
        UP = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.p.u.npy' \
                     % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED))
        UT = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.t.u.npy' \
                     % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED))
        UPE = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.pe.u.npy' \
                      % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED))
        UKE = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.ke.u.npy' \
                      % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED))
        UVOL = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.vol.u.npy' \
                       % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED))
        UNRHO = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.nrho.u.npy' \
                        % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED))
        # del PZENC, CZENC, VZENC, CDAT
        if VERBOSE:
            print('inlier selected z encodings loaded from file')
            print(100*'-')
    except:
        SLPZENC, UP, UT, UPE, UKE, UVOL, UNRHO = inlier_selection(PZENC.reshape(*SSHP3), CP, CT, CPE, CKE, CVOL, CNRHO, UNS)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.zenc.pca.prj.inl.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED), SLPZENC)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.p.u.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED), UP)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.t.u.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED), UT)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.pe.u.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED), UPE)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.ke.u.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED), UKE)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.vol.u.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED), UVOL)
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%d.%04d.nrho.u.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, AD, SEED), UNRHO)
        # del PZENC, CZENC, VZENC, CDAT
        if VERBOSE:
            print('inlier selected z encodings computed')
            print(100*'-')

    USHP0 = (NP, NT, UNS, 2, LD)
    USHP1 = (NP*NT*UNS, 2, LD)
    USHP2 = (NP, NT, UNS, LD)
    USHP3 = (NP*NT*UNS, LD)

    # reduction dictionary
    MNFLDS = {'pca':PCA(n_components=2),
              'kpca':KernelPCA(n_components=2, n_jobs=THREADS),
              'isomap':Isomap(n_components=2, n_jobs=THREADS),
              'lle':LocallyLinearEmbedding(n_components=2, n_jobs=THREADS),
              'tsne':TSNE(n_components=2, perplexity=UNS,
                          early_exaggeration=24, learning_rate=200, n_iter=1000,
                          verbose=VERBOSE, n_jobs=THREADS)}

    try:
        MSLZENC = np.load(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%s.%d.%04d.zenc.mfld.inl.npy' \
                          % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, MNFLD, AD, SEED))
        if VERBOSE:
            print('inlier selected z encoding manifold loaded from file')
            print(100*'-')
    except:
        MSLZENC = np.zeros((NP*NT*UNS, 2, 2))
        for i in range(ED):
            MSLZENC[:, i, :] = MNFLDS[MNFLD].fit_transform(SLPZENC[:, :, :, i, :].reshape(*USHP3))
        np.save(PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%s.%d.%04d.zenc.mfld.inl.npy' \
                % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, MNFLD, AD, SEED), MSLZENC)
        if VERBOSE:
            if MNFLD == 'tsne':
                print(100*'-')
            print('inlier selected z encoding manifold computed')
            print(100*'-')

    if PLOT:
        outpref = PREF+'.%04d.%s.%s.%s.%02d.%04d.%.0e.%04d.%s.%d.%04d' \
                  % (SNS, SCLR, OPT, LSS, LD, EP, LR, UNS, MNFLD, AD, SEED)
        for i in range(2):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.scatter(MSLZENC[:, i, 0], MSLZENC[:, i, 1], c=UT.reshape(-1),
                       cmap=plt.get_cmap('plasma'), s=64, alpha=0.5, edgecolors='')
            plt.xlabel('NL VAE %d0' % i)
            plt.ylabel('NL VAE %d1' % i)
            fig.savefig(outpref+'.vae.mnfld.prj.ld.%02d.png' % i)