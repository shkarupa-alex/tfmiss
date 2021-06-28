from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from nlpvocab import Vocabulary
from tensorflow.keras import activations, initializers, layers, utils
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.python.keras.utils import tf_utils
from tensorflow_text.python.ops.wordpiece_tokenizer import gen_wordpiece_tokenizer as wordpiece_tokenizer
from tensorflow_text.tools.wordpiece_vocab.wordpiece_tokenizer_learner_lib import learn as wordpiece_learner
from tfmiss.keras.layers import AdaptiveEmbedding, Reduction
from tfmiss import text as miss_text


@utils.register_keras_serializable(package='Miss')
class WordEmbedding(layers.Layer):
    UNK_MARK = '[UNK]'
    REP_CHAR = '\uFFFD'

    def __init__(self, vocabulary, output_dim, normalize_unicode='NFKC', lower_case=False, zero_digits=False,
                 max_len=None, reserved_words=None, embed_type='dense_auto', adapt_cutoff=None, adapt_factor=4,
                 embeddings_initializer='uniform', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=1, max_ndim=2, dtype='string')

        if not isinstance(vocabulary, list) or not all(map(lambda x: isinstance(x, str), vocabulary)):
            raise ValueError('Expected "vocabulary" to be a list of strings')
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError('Expected "vocabulary" to contain unique values')
        self.vocabulary = vocabulary

        self.output_dim = output_dim
        self.normalize_unicode = normalize_unicode
        self.lower_case = lower_case
        self.zero_digits = zero_digits

        if max_len is not None and max_len < 3:
            raise ValueError('Expected "max_len" to be None or greater then 2')
        self.max_len = max_len

        if reserved_words and len(reserved_words) != len(set(reserved_words)):
            raise ValueError('Expected "reserved_words" to contain unique values')
        self.reserved_words = reserved_words

        if embed_type not in {'dense_auto', 'dense_cpu', 'adapt'}:
            raise ValueError('Expected "embed_type" to be one of "dense_auto", "dense_cpu" or "adapt"')
        self.embed_type = embed_type

        self.adapt_cutoff = adapt_cutoff
        self.adapt_factor = adapt_factor
        self.embeddings_initializer = initializers.get(embeddings_initializer)

        all_reserved_words = [] if reserved_words is None else [r for r in reserved_words if self.UNK_MARK != r]
        self._reserved_words = [self.UNK_MARK] + all_reserved_words

        miss_reserved_words = [m for m in self._reserved_words if m not in vocabulary]
        if miss_reserved_words:
            tf.get_logger().warning('Vocabulary missed some reserved_words values: {}. '
                                    'This may indicate an error in vocabulary estimation'.format(miss_reserved_words))

        clean_vocab = [w for w in vocabulary if w not in self._reserved_words]
        self._vocabulary = self._reserved_words + clean_vocab

    def vocab(self, word_counts, **kwargs):
        if not word_counts:
            raise ValueError('Can\'t estimate vocabulary with empty word counter')
        if not all(map(lambda k: isinstance(k, str), word_counts.keys())):
            raise ValueError('Expected all words to be strings')

        word_counts = Vocabulary(word_counts)
        word_tokens = word_counts.tokens()
        adapt_words = self.adapt(word_tokens)
        if 1 == adapt_words.shape.rank:
            adapt_words = adapt_words[..., None]

        adapt_counts = Vocabulary()
        for adapts, word in zip(adapt_words, word_tokens):
            adapts = np.char.decode(adapts.numpy().reshape([-1]).astype('S'), 'utf-8')
            for adapt in adapts:
                adapt_counts[adapt] += word_counts[word]

        return adapt_counts

    @tf_utils.shape_type_conversion
    def build(self, input_shape=None):
        self.squeeze = False
        if 2 == len(input_shape):
            if 1 != input_shape[-1]:
                raise ValueError(
                    'Input 0 of layer {} is incompatible with the layer: if ndim=2 expected axis[-1]=1, found '
                    'axis[-1]={}. Full shape received: {}'.format(self.name, input_shape[-1], input_shape))

            self.squeeze = True
            input_shape = input_shape[:1]

        self.lookup = StringLookup(vocabulary=self._vocabulary, mask_token=None, oov_token=self.UNK_MARK)
        self.lookup.build(input_shape)

        if 'adapt' == self.embed_type:
            self.embed = AdaptiveEmbedding(
                self.adapt_cutoff, self.lookup.vocabulary_size(), self.output_dim, factor=self.adapt_factor,
                embeddings_initializer=self.embeddings_initializer)
        else:
            self.embed = layers.Embedding(
                self.lookup.vocabulary_size(), self.output_dim, embeddings_initializer=self.embeddings_initializer)
            if 'dense_auto' == self.embed_type:
                self.embed.build(input_shape)
            else:  # 'dense_cpu' == self.embed_type
                with tf.device('cpu:0'):
                    self.embed.build(input_shape)

        super().build(input_shape)

    def adapt(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype='string')

        if self.normalize_unicode:
            inputs = miss_text.normalize_unicode(inputs, form=self.normalize_unicode, skip=self._reserved_words)
        if self.lower_case:
            inputs = miss_text.lower_case(inputs, skip=self._reserved_words)
        if self.zero_digits:
            inputs = miss_text.zero_digits(inputs, skip=self._reserved_words)

        if self.max_len is not None:
            inputs_ = tf.stack([
                miss_text.sub_string(inputs, 0, self.max_len // 2, skip=self._reserved_words),
                tf.fill(tf.shape(inputs), self.REP_CHAR),
                miss_text.sub_string(inputs, -self.max_len // 2 + 1, -1, skip=self._reserved_words)],
                axis=-1)
            inputs_ = tf.strings.reduce_join(inputs_, axis=-1)
            sizes = tf.strings.length(inputs, unit='UTF8_CHAR')
            inputs = tf.where(sizes > self.max_len, inputs_, inputs)

        return inputs

    def call(self, inputs, **kwargs):
        if self.squeeze:
            # Workaround for Sequential model test
            inputs = tf.squeeze(inputs, axis=-1)

        adapts = self.adapt(inputs)
        indices = self.lookup(adapts)
        outputs = self.embed(indices)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary': self.vocabulary,
            'output_dim': self.output_dim,
            'normalize_unicode': self.normalize_unicode,
            'lower_case': self.lower_case,
            'zero_digits': self.zero_digits,
            'max_len': self.max_len,
            'reserved_words': self.reserved_words,
            'embed_type': self.embed_type,
            'adapt_cutoff': self.adapt_cutoff,
            'adapt_factor': self.adapt_factor,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer)
        })

        return config


@tf.keras.utils.register_keras_serializable(package='Miss')
class CharNgramEmbedding(WordEmbedding):
    BOW_MARK = '<'
    EOW_MARK = '>'

    def __init__(self, vocabulary, output_dim, minn=3, maxn=5, itself='always', reduction='mean', **kwargs):
        super().__init__(vocabulary, output_dim, **kwargs)

        self.minn = minn
        self.maxn = maxn
        self.itself = itself
        self.reduction = reduction

    @tf_utils.shape_type_conversion
    def build(self, input_shape=None):
        self.reduce = Reduction(self.reduction)

        super().build(input_shape)

    def adapt(self, inputs):
        wraps = miss_text.wrap_with(inputs, self.BOW_MARK, self.EOW_MARK, skip=self._reserved_words)
        adapts = super().adapt(wraps)
        ngrams = miss_text.char_ngrams(adapts, self.minn, self.maxn, self.itself, skip=self._reserved_words)

        return ngrams

    def call(self, inputs, **kwargs):
        embeds = super().call(inputs)
        outputs = self.reduce(embeds)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'minn': self.minn,
            'maxn': self.maxn,
            'itself': self.itself,
            'reduction': self.reduction
        })

        return config


@tf.keras.utils.register_keras_serializable(package='Miss')
class CharBpeEmbedding(WordEmbedding):
    UNK_CHAR = '##[UNK]'

    def __init__(self, vocabulary, output_dim, reduction='mean', vocab_size=32000, upper_thresh=None, lower_thresh=2,
                 num_iterations=4, max_tokens=-1, max_chars=1000, slack_ratio=0.05, include_joiner=True,
                 joiner_prefix='##', reserved_words=None, **kwargs):
        _reserved_words = [self.UNK_CHAR]
        _reserved_words += [] if reserved_words is None else [r for r in reserved_words if r not in _reserved_words]

        super().__init__(vocabulary, output_dim, reserved_words=_reserved_words, **kwargs)

        self.reduction = reduction
        self.vocab_size = vocab_size
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh
        self.num_iterations = num_iterations
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.slack_ratio = slack_ratio
        self.include_joiner = include_joiner
        self.joiner_prefix = joiner_prefix

    def vocab(self, word_counts, **kwargs):
        if not word_counts:
            raise ValueError('Can\'t estimate vocabulary with empty word counter')
        if not all(map(lambda k: isinstance(k, str), word_counts.keys())):
            raise ValueError('Expected all words to be strings')

        word_counts = Vocabulary(word_counts)
        sub_words = wordpiece_learner(
            word_counts,
            vocab_size=self.vocab_size,
            reserved_tokens=self._reserved_words,
            upper_thresh=self.upper_thresh,
            lower_thresh=self.lower_thresh,
            num_iterations=self.num_iterations,
            max_input_tokens=self.max_tokens,
            max_token_length=self.max_len or 9999,
            max_unique_chars=self.max_chars,
            slack_ratio=self.slack_ratio,
            include_joiner_token=self.include_joiner,
            joiner=self.joiner_prefix)

        word_tokens = word_counts.tokens()
        self_config = self.get_config()
        self_config.update({'vocabulary': sub_words})
        self_copy = self.from_config(self_config)

        adapt_words = self_copy.adapt(word_tokens)
        adapt_counts = Vocabulary()
        for adapts, word in zip(adapt_words, word_tokens):
            adapts = np.char.decode(adapts.numpy().reshape([-1]).astype('S'), 'utf-8')
            for adapt in adapts:
                adapt_counts[adapt] += word_counts[word]

        return adapt_counts

    @tf_utils.shape_type_conversion
    def build(self, input_shape=None):
        self.reduce = Reduction(self.reduction)

        super().build(input_shape)

    def adapt(self, inputs):
        if not self.built:
            self.build([None])
        adapts = super().adapt(inputs)

        values, row_splits, starts, ends = wordpiece_tokenizer.wordpiece_tokenize_with_offsets(
            input_values=adapts,
            vocab_lookup_table=self.lookup._table.resource_handle,
            suffix_indicator=self.joiner_prefix,
            use_unknown_token=True,
            max_bytes_per_word=(self.max_len or 9999) * 4,
            max_chars_per_token=0,
            unknown_token=self.UNK_MARK,
            split_unknown_characters=True,
            output_row_partition_type='row_splits')
        subwords = tf.RaggedTensor.from_row_splits(values, row_splits)

        return subwords

    def call(self, inputs, **kwargs):
        embeds = super().call(inputs)
        outputs = self.reduce(embeds)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction': self.reduction,
            'vocab_size': self.vocab_size,
            'upper_thresh': self.upper_thresh,
            'lower_thresh': self.lower_thresh,
            'num_iterations': self.num_iterations,
            'max_tokens': self.max_tokens,
            'max_chars': self.max_chars,
            'slack_ratio': self.slack_ratio,
            'include_joiner': self.include_joiner,
            'joiner_prefix': self.joiner_prefix,
        })

        return config


@tf.keras.utils.register_keras_serializable(package='Miss')
class CharCnnEmbedding(WordEmbedding):
    BOW_MARK = '[BOW]'
    EOW_MARK = '[EOW]'

    def __init__(self, vocabulary, output_dim, filters=(32, 32, 64, 128, 256, 512, 1024),
                 kernels=(1, 2, 3, 4, 5, 6, 7), char_dim=16, activation='tanh', highways=2,
                 embeddings_initializer=initializers.random_uniform(-1., 1.), max_len=50, reserved_words=None,
                 **kwargs):

        _reserved_words = [self.BOW_MARK, self.EOW_MARK]
        _reserved_words += [] if reserved_words is None else [r for r in reserved_words if r not in _reserved_words]
        _max_len = None if max_len is None else max_len - 2
        super().__init__(
            vocabulary, output_dim, embeddings_initializer=embeddings_initializer, max_len=_max_len,
            reserved_words=_reserved_words, **kwargs)
        self.max_len = _max_len

        if not filters or not isinstance(filters, list) or not all(map(lambda x: isinstance(x, int), filters)):
            raise ValueError('Expected "filters" argument to be a list of integers')
        if not kernels or not isinstance(kernels, list) or not all(map(lambda x: isinstance(x, int), kernels)):
            raise ValueError('Expected "kernels" argument to be a list of integers')
        if len(filters) != len(kernels):
            raise ValueError('Sizes of "filters" and "kernels" should be equal')
        self.filters = filters
        self.kernels = kernels

        self.char_dim = char_dim
        self.activation = activations.get(activation)
        self.highways = highways

    def build(self, input_shape=None):
        self._reserved_words = [self.UNK_MARK, self.BOW_MARK, self.EOW_MARK] + (
            [] if self.reserved_words is None else self.reserved_words)

        def _kernel_init(kernel_size):
            if 'tanh' == activations.serialize(self.activation):
                stddev = np.sqrt(1. / (kernel_size * self.char_dim))

                return initializers.random_normal(mean=0., stddev=stddev)

            if 'relu' == activations.serialize(self.activation):
                return 'random_uniform'

            return 'glorot_uniform'

        self.conv = [
            layers.Conv1D(f, k, padding='same', kernel_initializer=_kernel_init(k))
            for f, k in zip(self.filters, self.kernels)]
        self.pool = layers.GlobalMaxPool1D()
        self.act = layers.Activation(self.activation)
        self.high = [Highway() for _ in range(self.highways)]
        self.proj = None if sum(self.filters) == self.output_dim else layers.Dense(self.output_dim)

        super().build(input_shape)

    def adapt(self, inputs):
        adapts = super().adapt(inputs)
        chars = miss_text.split_chars(adapts, skip=self._reserved_words)

        bos = tf.fill([chars.nrows(), 1], self.BOW_MARK)
        eos = tf.fill([chars.nrows(), 1], self.EOW_MARK)
        wraps = tf.concat([bos, chars, eos], axis=1)

        return wraps

    def call(self, inputs, **kwargs):
        embeds = super().call(inputs)
        embeds = embeds.to_tensor()

        convs = [c(embeds) for c in self.conv]
        convs = [self.pool(c) for c in convs]
        convs = [self.act(c) for c in convs]
        convs = layers.concatenate(convs)

        outputs = convs
        for h in self.high:
            outputs = h(outputs)

        if self.proj is not None:
            outputs = self.proj(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': None if self.max_len is None else self.max_len + 2,
            'filters': self.filters,
            'kernels': self.kernels,
            'char_dim': self.char_dim,
            'activation': activations.serialize(self.activation),
            'highways': self.highways
        })

        return config


@tf.keras.utils.register_keras_serializable(package='Miss')
class Highway(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=2)

    def build(self, input_shape=None):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=max(2, len(input_shape)), axes={-1: channels})

        kernel_initializer = initializers.random_normal(mean=0., stddev=np.sqrt(1. / channels))
        self.carry = layers.Dense(
            channels,
            kernel_initializer=kernel_initializer,
            bias_initializer=initializers.constant(-2.),
            activation='sigmoid')
        self.transform = layers.Dense(
            channels,
            kernel_initializer=kernel_initializer,
            activation='relu')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        carry_gate = self.carry(inputs)
        transform_gate = self.transform(inputs)
        outputs = carry_gate * transform_gate + (1. - carry_gate) * inputs

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
