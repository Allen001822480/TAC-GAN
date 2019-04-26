#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Thu Apr 25 19:18:05 2019@author: osamuyanagano"""import osfrom os.path import joinimport pickle as pklimport mathimport numpyfrom collections import OrderedDict, defaultdictimport theanoimport theano.tensor as tensorimport nltkfrom scipy.linalg import normfrom nltk.tokenize import word_tokenizeimport tensorflow as tfimport modelimport scipy#-----------------------------------------------------------------------------## Specify model and table locations here#-----------------------------------------------------------------------------#path_to_models = 'Data/skipthoughts/'path_to_tables = 'Data/skipthoughts/'#-----------------------------------------------------------------------------#path_to_umodel = path_to_models + 'uni_skip.npz'path_to_bmodel = path_to_models + 'bi_skip.npz'#TextEncodingdef load_model(path_to_bmodel = 'Data/skipthoughts/bi_skip.npz' ,path_to_umodel = 'Data/skipthoughts/uni_skip.npz', path_to_models = 'Data/skipthoughts/', path_to_tables = 'Data/skipthoughts/', profile = False):    """    Load the model with saved tables    """    # Load model options    print('Loading model parameters...')    with open('%s.pkl'%path_to_umodel, 'rb') as f:        uoptions = pkl.load(f)    with open('%s.pkl'%path_to_bmodel, 'rb') as f:        boptions = pkl.load(f)    # Load parameters    uparams = init_params(uoptions)    uparams = load_params(path_to_umodel, uparams)    utparams = init_tparams(uparams)    bparams = init_params_bi(boptions)    bparams = load_params(path_to_bmodel, bparams)    btparams = init_tparams(bparams)    # Extractor functions    print('Compiling encoders...')    embedding, x_mask, ctxw2v = build_encoder(utparams, uoptions)    f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')    embedding, x_mask, ctxw2v = build_encoder_bi(btparams, boptions)    f_w2v2 = theano.function([embedding, x_mask], ctxw2v, name='f_w2v2')    # Tables    print('Loading tables...')    utable, btable = load_tables()    # Store everything we need in a dictionary    print('Packing up...')    model = {}    model['uoptions'] = uoptions    model['boptions'] = boptions    model['utable'] = utable    model['btable'] = btable    model['f_w2v'] = f_w2v    model['f_w2v2'] = f_w2v2    return modeldef load_tables():    """    Load the tables    """    words = []    utable = numpy.load(path_to_tables + 'utable.npy', encoding='bytes',allow_pickle=True)    btable = numpy.load(path_to_tables + 'btable.npy', encoding='bytes',allow_pickle=True)    f = open(path_to_tables + 'dictionary.txt', 'rb')    for line in f:        words.append(line.decode('utf-8').strip())    f.close()    utable = OrderedDict(zip(words, utable))    btable = OrderedDict(zip(words, btable))    return utable, btabledef encode(model, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):    """    Encode sentences in the list X. Each entry will return a vector    """    # first, do preprocessing    X = preprocess(X)    # word dictionary and init    d = defaultdict(lambda : 0)    for w in model['utable'].keys():        d[w] = 1    ufeatures = numpy.zeros((len(X), model['uoptions']['dim']), dtype='float32')    bfeatures = numpy.zeros((len(X), 2 * model['boptions']['dim']), dtype='float32')    # length dictionary    ds = defaultdict(list)    captions = [s.split() for s in X]    for i,s in enumerate(captions):        ds[len(s)].append(i)    # Get features. This encodes by length, in order to avoid wasting computation    for k in ds.keys():        if verbose:            print(k)        numbatches = len(ds[k]) / batch_size + 1        for minibatch in range(int(numbatches)):            caps = ds[k][int(minibatch)::int(numbatches)]            if use_eos:                uembedding = numpy.zeros((k+1, len(caps), model['uoptions']['dim_word']), dtype='float32')                bembedding = numpy.zeros((k+1, len(caps), model['boptions']['dim_word']), dtype='float32')            else:                uembedding = numpy.zeros((k, len(caps), model['uoptions']['dim_word']), dtype='float32')                bembedding = numpy.zeros((k, len(caps), model['boptions']['dim_word']), dtype='float32')            for ind, c in enumerate(caps):                caption = captions[c]                for j in range(len(caption)):                    if d[caption[j]] > 0:                        uembedding[j,ind] = model['utable'][caption[j]]                        bembedding[j,ind] = model['btable'][caption[j]]                    else:                        uembedding[j,ind] = model['utable']['UNK']                        bembedding[j,ind] = model['btable']['UNK']                if use_eos:                    uembedding[-1,ind] = model['utable']['<eos>']                    bembedding[-1,ind] = model['btable']['<eos>']            if use_eos:                uff = model['f_w2v'](uembedding, numpy.ones((len(caption)+1,len(caps)), dtype='float32'))                bff = model['f_w2v2'](bembedding, numpy.ones((len(caption)+1,len(caps)), dtype='float32'))            else:                uff = model['f_w2v'](uembedding, numpy.ones((len(caption),len(caps)), dtype='float32'))                bff = model['f_w2v2'](bembedding, numpy.ones((len(caption),len(caps)), dtype='float32'))            if use_norm:                for j in range(len(uff)):                    uff[j] /= norm(uff[j])                    bff[j] /= norm(bff[j])            for ind, c in enumerate(caps):                ufeatures[c] = uff[ind]                bfeatures[c] = bff[ind]    features = numpy.c_[ufeatures, bfeatures]    return featuresdef preprocess(text):    """    Preprocess text for encoder    """    X = []    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')    for t in text:        sents = sent_detector.tokenize(t)        result = ''        for s in sents:            tokens = word_tokenize(s)            result += ' ' + ' '.join(tokens)        X.append(result)    return Xdef nn(model, text, vectors, query, k=5):    """    Return the nearest neighbour sentences to query    text: list of sentences    vectors: the corresponding representations for text    query: a string to search    """    qf = encode(model, [query])    qf /= norm(qf)    scores = numpy.dot(qf, vectors.T).flatten()    sorted_args = numpy.argsort(scores)[::-1]    sentences = [text[a] for a in sorted_args[:k]]    print('QUERY: ' + query)    print('NEAREST: ')    for i, s in enumerate(sentences):        print(s, sorted_args[i])def word_features(table):    """    Extract word features into a normalized matrix    """    features = numpy.zeros((len(table), 620), dtype='float32')    keys = table.keys()    for i in range(len(table)):        f = table[keys[i]]        features[i] = f / norm(f)    return featuresdef nn_words(table, wordvecs, query, k=10):    """    Get the nearest neighbour words    """    keys = table.keys()    qf = table[query]    scores = numpy.dot(qf, wordvecs.T).flatten()    sorted_args = numpy.argsort(scores)[::-1]    words = [keys[a] for a in sorted_args[:k]]    print('QUERY: ' + query)    print('NEAREST: ')    for i, w in enumerate(words):        print(w)def _p(pp, name):    """    make prefix-appended name    """    return '%s_%s'%(pp, name)def init_tparams(params):    """    initialize Theano shared variables according to the initial parameters    """    tparams = OrderedDict()    for kk, pp in params.items():        tparams[kk] = theano.shared(params[kk], name=kk)    return tparamsdef load_params(path, params):    """    load parameters    """    pp = numpy.load(path)    for kk, vv in params.items():        if kk not in pp:            continue        params[kk] = pp[kk]    return params# layers: 'name': ('parameter initializer', 'feedforward')layers = {'gru': ('param_init_gru', 'gru_layer')}def get_layer(name):    fns = layers[name]    return (eval(fns[0]), eval(fns[1]))def init_params(options):    """    initialize all parameters needed for the encoder    """    params = OrderedDict()    # embedding    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])    # encoder: GRU    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',                                              nin=options['dim_word'], dim=options['dim'])    return paramsdef init_params_bi(options):    """    initialize all paramters needed for bidirectional encoder    """    params = OrderedDict()    # embedding    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])    # encoder: GRU    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',                                              nin=options['dim_word'], dim=options['dim'])    params = get_layer(options['encoder'])[0](options, params, prefix='encoder_r',                                              nin=options['dim_word'], dim=options['dim'])    return paramsdef build_encoder(tparams, options):    """    build an encoder, given pre-computed word embeddings    """    # word embedding (source)    embedding = tensor.tensor3('embedding', dtype='float32')    x_mask = tensor.matrix('x_mask', dtype='float32')    # encoder    proj = get_layer(options['encoder'])[1](tparams, embedding, options,                                            prefix='encoder',                                            mask=x_mask)    ctx = proj[0][-1]    return embedding, x_mask, ctxdef build_encoder_bi(tparams, options):    """    build bidirectional encoder, given pre-computed word embeddings    """    # word embedding (source)    embedding = tensor.tensor3('embedding', dtype='float32')    embeddingr = embedding[::-1]    x_mask = tensor.matrix('x_mask', dtype='float32')    xr_mask = x_mask[::-1]    # encoder    proj = get_layer(options['encoder'])[1](tparams, embedding, options,                                            prefix='encoder',                                            mask=x_mask)    projr = get_layer(options['encoder'])[1](tparams, embeddingr, options,                                             prefix='encoder_r',                                             mask=xr_mask)    ctx = tensor.concatenate([proj[0][-1], projr[0][-1]], axis=1)    return embedding, x_mask, ctx# some utilitiesdef ortho_weight(ndim):    W = numpy.random.randn(ndim, ndim)    u, s, v = numpy.linalg.svd(W)    return u.astype('float32')def norm_weight(nin,nout=None, scale=0.1, ortho=True):    if nout == None:        nout = nin    if nout == nin and ortho:        W = ortho_weight(nin)    else:        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))    return W.astype('float32')def param_init_gru(options, params, prefix='gru', nin=None, dim=None):    """    parameter init for GRU    """    if nin == None:        nin = options['dim_proj']    if dim == None:        dim = options['dim_proj']    W = numpy.concatenate([norm_weight(nin,dim),                           norm_weight(nin,dim)], axis=1)    params[_p(prefix,'W')] = W    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')    U = numpy.concatenate([ortho_weight(dim),                           ortho_weight(dim)], axis=1)    params[_p(prefix,'U')] = U    Wx = norm_weight(nin, dim)    params[_p(prefix,'Wx')] = Wx    Ux = ortho_weight(dim)    params[_p(prefix,'Ux')] = Ux    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')    return paramsdef gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):    """    Forward pass through GRU layer    """    nsteps = state_below.shape[0]    if state_below.ndim == 3:        n_samples = state_below.shape[1]    else:        n_samples = 1    dim = tparams[_p(prefix,'Ux')].shape[1]    if mask == None:        mask = tensor.alloc(1., state_below.shape[0], 1)    def _slice(_x, n, dim):        if _x.ndim == 3:            return _x[:, :, n*dim:(n+1)*dim]        return _x[:, n*dim:(n+1)*dim]    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]    tparams[_p(prefix, 'U')]    tparams[_p(prefix, 'Ux')]    def _step_slice(m_, x_, xx_, h_, U, Ux):        preact = tensor.dot(h_, U)        preact += x_        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))        preactx = tensor.dot(h_, Ux)        preactx = preactx * r        preactx = preactx + xx_        h = tensor.tanh(preactx)        h = u * h_ + (1. - u) * h        h = m_[:,None] * h + (1. - m_)[:,None] * h_        return h    seqs = [mask, state_below_, state_belowx]    _step = _step_slice    rval, updates = theano.scan(_step,                                sequences=seqs,                                outputs_info = [tensor.alloc(0., n_samples, dim)],                                non_sequences = [tparams[_p(prefix, 'U')],                                                 tparams[_p(prefix, 'Ux')]],                                name=_p(prefix, '_layers'),                                n_steps=nsteps,                                profile=False,                                strict=True)    rval = [rval]    return rvaldef encode_text(str_captions,data_dir = "Data"):    skp_model = load_model()    captions = str_captions.split('\n')    encoded_captions = {}    encoded_captions['features'] = encode(skp_model, captions)    dump_path = os.path.join(data_dir, 'enc_text.pkl')    pkl.dump(encoded_captions,                open(dump_path, "wb"))    print('Finished extracting Skip-Thought vectors of the given text '          'descriptions')#ImageGenerationdef generate_images(z_dim, t_dim, batch_size = 256, image_size = 128, gf_dim = 64, df_dim = 64, caption_vector_length = 4800, n_classes = 102,                    data_dir = 'Data', learning_rate = 0.0002, beta1 = 0.5, images_per_caption = 1, data_set = 'flowers', checkpoints_dir = 'training/TAC-GAN'):    datasets_root_dir = data_dir    loaded_data = load_training_data(datasets_root_dir, data_set, n_classes)    model_options = {        'z_dim': z_dim,        't_dim': t_dim,        'batch_size': batch_size,        'image_size': image_size,        'gf_dim': gf_dim,        'df_dim': df_dim,        'caption_vector_length': caption_vector_length,        'n_classes': n_classes    }    gan = model.GAN(model_options)    input_tensors, variables, loss, outputs, checks = gan.build_model()    sess = tf.InteractiveSession()    init = tf.reset_default_graph()    sess.run(init)    saver = tf.train.Saver(max_to_keep=10000)    print('Trying to resume model from ' +          str(tf.train.latest_checkpoint(checkpoints_dir)))    if tf.train.latest_checkpoint(checkpoints_dir) is not None:        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))        print('Successfully loaded model from ')    else:        print('Could not load checkpoints. Please provide a valid path to'              ' your checkpoints directory')        exit()    print('Starting to generate images from text descriptions.')    for sel_i, text_cap in enumerate(loaded_data['text_caps']['features']):        print('Text idx: {}\nRaw Text: {}\n'.format(sel_i, text_cap))        captions_1, image_files_1, image_caps_1, image_ids_1,\        image_caps_ids_1 = get_caption_batch(loaded_data, data_dir,                         dataset=data_set, batch_size=batch_size)        captions_1[batch_size-1, :] = text_cap        for z_i in range(images_per_caption):            z_noise = numpy.random.uniform(-1, 1, [batch_size, 100])            val_feed = {                input_tensors['t_real_caption'].name: captions_1,                input_tensors['t_z'].name: z_noise,                input_tensors['t_training'].name: True            }            val_gen = sess.run(                [outputs['generator']],                feed_dict=val_feed)            dump_dir = os.path.join(data_dir,                                    'images_generated_from_text')            save_distributed_image_batch(dump_dir, val_gen, sel_i, z_i,                                         batch_size)    print('Finished generating images from text description')    sess.close()def load_training_data(data_dir, data_set, caption_vector_length, n_classes = 102):    if data_set == 'flowers':        flower_str_captions = pkl.load(            open(join(data_dir, 'flowers_caps.pkl'), "rb"))        img_classes = pkl.load(            open(join(data_dir, 'flower_tc.pkl'), "rb"))        flower_enc_captions = pkl.load(            open(join(data_dir, 'flower_tv.pkl'), "rb"))        # h1 = h5py.File(join(data_dir, 'flower_tc.hdf5'))        tr_image_ids = pkl.load(            open(join(data_dir, 'train_ids.pkl'), "rb"))        val_image_ids = pkl.load(            open(join(data_dir, 'val_ids.pkl'), "rb"))        caps_new = pkl.load(            open(join('Data', 'enc_text.pkl'), "rb"))        # n_classes = n_classes        max_caps_len = 4800        tr_n_imgs = len(tr_image_ids)        val_n_imgs = len(val_image_ids)        return {            'image_list': tr_image_ids,            'captions': flower_enc_captions,            'data_length': tr_n_imgs,            'classes': img_classes,            'n_classes': n_classes,            'max_caps_len': max_caps_len,            'val_img_list': val_image_ids,            'val_captions': flower_enc_captions,            'val_data_len': val_n_imgs,            'str_captions': flower_str_captions,            'text_caps': caps_new        }    else:        raise Exception('This dataset has not been handeled yet. '                         'Contributions are welcome.')def save_distributed_image_batch(data_dir, generated_images, sel_i, z_i,                                 batch_size=64):    generated_images = numpy.squeeze(generated_images)    folder_name = str(sel_i)    image_dir = join(data_dir, folder_name)    if not os.path.exists(image_dir):        os.makedirs(image_dir)    fake_image_255 = generated_images[batch_size-1]    scipy.misc.imsave(join(image_dir, '{}.jpg'.format(z_i)),                          fake_image_255)def get_caption_batch(loaded_data, data_dir, dataset='flowers', batch_size=64):    captions = numpy.zeros((batch_size, loaded_data['max_caps_len']))    batch_idx = numpy.random.randint(0, loaded_data['data_length'],                                  size=batch_size)    image_ids = numpy.take(loaded_data['image_list'], batch_idx)    image_files = []    image_caps = []    image_caps_ids = []    for idx, image_id in enumerate(image_ids):        image_file = join(data_dir, dataset, 'jpg' + image_id)        random_caption = numpy.random.randint(0, 4)        image_caps_ids.append(random_caption)        captions[idx, :] = \            loaded_data['captions'][image_id][random_caption][            0:loaded_data['max_caps_len']]        image_caps.append(loaded_data['captions']                          [image_id][random_caption])        image_files.append(image_file)    return captions, image_files, image_caps, image_ids, image_caps_idsdef combine_normalized_images(generated_images):    num = generated_images.shape[0]    width = int(math.sqrt(num))    height = int(math.ceil(float(num) / width))    shape = generated_images.shape[1:]    image = numpy.zeros((height * shape[0], width * shape[1], shape[2]),dtype=generated_images.dtype)    for index, img in enumerate(generated_images):        i = int(index / width)        j = index % width        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img    return imagedef tac_gan_pred(text):    encode_text(text)    generate_images(t_dim = 100,z_dim = 100)if __name__ == '__main__':    tac_gan_pred(text = 'This is a yellow flower.')