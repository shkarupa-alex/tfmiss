from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from nlpvocab import Vocabulary
from keras import activations, initializers, layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfmiss.keras.layers import AdaptiveEmbedding, Reduction
from tfmiss import text as miss_text


@register_keras_serializable(package='Miss')
class WordEmbedding(layers.Layer):
    UNK_MARK = '[UNK]'
    REP_CHAR = '\uFFFD'

    def __init__(self, vocabulary=None, output_dim=None, normalize_unicode='NFKC', lower_case=False, zero_digits=False,
                 max_len=None, reserved_words=None, embed_type='dense_auto', adapt_cutoff=None, adapt_factor=4,
                 with_prep=False, embeddings_initializer='uniform', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=1, dtype='string' if with_prep else 'int64')
        self._supports_ragged_inputs = True

        if vocabulary is not None:
            if not isinstance(vocabulary, list) or not all(map(lambda x: isinstance(x, str), vocabulary)):
                raise ValueError('Expecting "vocabulary" to be a list of strings')
            if len(vocabulary) != len(set(vocabulary)):
                raise ValueError('Expecting "vocabulary" to contain unique values')
        self.vocabulary = vocabulary

        if output_dim is not None and output_dim < 1:
            raise ValueError('Expecting "output_dim" to be None or greater then 0')
        self.output_dim = output_dim

        self.normalize_unicode = normalize_unicode
        self.lower_case = lower_case
        self.zero_digits = zero_digits

        if max_len is not None and max_len < 3:
            raise ValueError('Expecting "max_len" to be None or greater then 2')
        self.max_len = max_len

        if reserved_words is not None:
            if not isinstance(reserved_words, list) or not all(map(lambda x: isinstance(x, str), reserved_words)):
                raise ValueError('Expecting "reserved_words" to be a list of strings')
            if len(reserved_words) != len(set(reserved_words)):
                raise ValueError('Expecting "reserved_words" to contain unique values')
        self.reserved_words = reserved_words

        if embed_type not in {'dense_auto', 'dense_cpu', 'adapt'}:
            raise ValueError('Expecting "embed_type" to be one of "dense_auto", "dense_cpu" or "adapt"')
        self.embed_type = embed_type

        self.adapt_cutoff = adapt_cutoff
        self.adapt_factor = adapt_factor
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.with_prep = with_prep

        self._reserved_words = [] if reserved_words is None else reserved_words
        self._reserved_words = [self.UNK_MARK] + [r for r in self._reserved_words if self.UNK_MARK != r]

        self._vocabulary = [] if vocabulary is None else vocabulary
        self._vocabulary = self._reserved_words + [v for v in self._vocabulary if v not in self._reserved_words]

        self._lookup = None

    def vocab(self, word_counts, **kwargs):
        if not word_counts:
            raise ValueError('Can\'t estimate vocabulary with empty word counter')
        if not all(map(lambda k: isinstance(k, str), word_counts.keys())):
            raise ValueError('Expecting all words to be strings')
        if not all(map(lambda k: isinstance(k, int), word_counts.values())):
            raise ValueError('Expecting all frequencies to be integers')

        word_counts = Vocabulary(word_counts)
        word_tokens = word_counts.tokens()
        adapt_words = self.adapt(word_tokens)
        if 1 == adapt_words.shape.rank:
            adapt_words = adapt_words[..., None]

        adapt_counts = Vocabulary()
        for adapts, word in zip(adapt_words, word_tokens):
            adapts = np.char.decode(adapts.numpy().ravel().astype('S'), 'utf-8')
            for adapt in adapts:
                adapt_counts[adapt] += word_counts[word]

        return adapt_counts

    def adapt(self, inputs):
        outputs = inputs

        if self.normalize_unicode:
            outputs = miss_text.normalize_unicode(outputs, form=self.normalize_unicode, skip=self._reserved_words)
        if self.lower_case:
            outputs = miss_text.lower_case(outputs, skip=self._reserved_words)
        if self.zero_digits:
            outputs = miss_text.zero_digits(outputs, skip=self._reserved_words)

        if self.max_len is not None:
            values = outputs
            if isinstance(outputs, tf.RaggedTensor):
                values = outputs.flat_values

            lengths = tf.strings.length(values, unit='UTF8_CHAR')

            cutouts = tf.stack([
                miss_text.sub_string(values, 0, self.max_len // 2, skip=self._reserved_words),
                tf.fill(tf.shape(values), self.REP_CHAR),
                miss_text.sub_string(values, -self.max_len // 2 + 1, -1, skip=self._reserved_words)], axis=-1)
            cutouts = tf.strings.reduce_join(cutouts, axis=-1)
            cutouts = tf.where(lengths > self.max_len, cutouts, values)

            if isinstance(outputs, tf.RaggedTensor):
                outputs = outputs.with_flat_values(cutouts)
            else:
                outputs = cutouts

        return outputs

    def lookup(self, inputs=None):
        if self._lookup is None:
            if not set(self._vocabulary) - set(self._reserved_words):
                raise ValueError('Can\'t make lookup with empty vocabulary')

            self._lookup = layers.StringLookup(vocabulary=self._vocabulary, mask_token=None, oov_token=self.UNK_MARK)

        if inputs is None:
            return None

        return self._lookup(inputs)

    def preprocess(self, inputs):
        adapts = self.adapt(inputs)
        indices = self.lookup(adapts)

        return indices

    @shape_type_conversion
    def build(self, input_shape):
        if not self.output_dim:
            raise ValueError('Can\'t build embedding matrix without output_dim')

        self.lookup()  # Init lookup table

        if 'adapt' == self.embed_type:
            self.embed = AdaptiveEmbedding(
                self.adapt_cutoff, self._lookup.vocabulary_size(), self.output_dim, factor=self.adapt_factor,
                embeddings_initializer=self.embeddings_initializer)
        else:
            self.embed = layers.Embedding(
                self._lookup.vocabulary_size(), self.output_dim, embeddings_initializer=self.embeddings_initializer)
            if 'dense_cpu' == self.embed_type:
                with tf.device('cpu:0'):
                    self.embed.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.with_prep:
            inputs = self.preprocess(inputs)

        return self.embed(inputs)

    @shape_type_conversion
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
            'with_prep': self.with_prep,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer)
        })

        return config


@register_keras_serializable(package='Miss')
class NgramEmbedding(WordEmbedding):
    BOW_MARK = '<'
    EOW_MARK = '>'

    def __init__(self, vocabulary=None, output_dim=None, minn=3, maxn=5, itself='always', reduction='mean',
                 normalize_unicode='NFKC', lower_case=False, zero_digits=False, max_len=None, reserved_words=None,
                 embed_type='dense_auto', adapt_cutoff=None, adapt_factor=4, with_prep=False,
                 embeddings_initializer='uniform', **kwargs):
        super().__init__(vocabulary=vocabulary, output_dim=output_dim, normalize_unicode=normalize_unicode,
                         lower_case=lower_case, zero_digits=zero_digits, max_len=max_len, reserved_words=reserved_words,
                         embed_type=embed_type, adapt_cutoff=adapt_cutoff, adapt_factor=adapt_factor,
                         with_prep=with_prep, embeddings_initializer=embeddings_initializer, **kwargs)

        self.minn = minn
        self.maxn = maxn
        self.itself = itself
        self.reduction = reduction

    def adapt(self, inputs):
        adapts = super().adapt(inputs)
        wraps = miss_text.wrap_with(adapts, self.BOW_MARK, self.EOW_MARK, skip=self._reserved_words)
        ngrams = miss_text.char_ngrams(wraps, self.minn, self.maxn, self.itself, skip=self._reserved_words)

        return ngrams

    @shape_type_conversion
    def build(self, input_shape=None):
        self.reduce = Reduction(self.reduction)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not self.with_prep and not isinstance(inputs, tf.RaggedTensor):
            raise ValueError('Expecting "inputs to be "RaggedTensor" instance.')

        embeds = super().call(inputs)
        outputs = self.reduce(embeds)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        prep_shape = input_shape
        if not self.with_prep:
            prep_shape = input_shape[:-1]

        return prep_shape + (self.output_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'minn': self.minn,
            'maxn': self.maxn,
            'itself': self.itself,
            'reduction': self.reduction
        })

        return config


@register_keras_serializable(package='Miss')
class BpeEmbedding(WordEmbedding):
    UNK_CHAR = '##[UNK]'

    def __init__(self, vocabulary=None, output_dim=None, vocab_size=32000, upper_thresh=None, lower_thresh=2,
                 num_iterations=4, max_tokens=-1, max_chars=1000, slack_ratio=0.05, include_joiner=True,
                 joiner_prefix='##', reduction='mean', normalize_unicode='NFKC', lower_case=False, zero_digits=False,
                 max_len=None, reserved_words=None, embed_type='dense_auto', adapt_cutoff=None, adapt_factor=4,
                 with_prep=False, embeddings_initializer='uniform', **kwargs):

        _reserved_words = [self.UNK_CHAR]
        _reserved_words += [] if reserved_words is None else [r for r in reserved_words if r not in _reserved_words]

        super().__init__(vocabulary=vocabulary, output_dim=output_dim, normalize_unicode=normalize_unicode,
                         lower_case=lower_case, zero_digits=zero_digits, max_len=max_len,
                         reserved_words=_reserved_words, embed_type=embed_type, adapt_cutoff=adapt_cutoff,
                         adapt_factor=adapt_factor, with_prep=with_prep, embeddings_initializer=embeddings_initializer,
                         **kwargs)

        self.vocab_size = vocab_size
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh
        self.num_iterations = num_iterations
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.slack_ratio = slack_ratio
        self.include_joiner = include_joiner
        self.joiner_prefix = joiner_prefix
        self.reduction = reduction

    def vocab(self, word_counts, **kwargs):
        if not word_counts:
            raise ValueError('Can\'t estimate vocabulary with empty word counter')
        if not all(map(lambda k: isinstance(k, str), word_counts.keys())):
            raise ValueError('Expecting all words to be strings')

        word_counts = Vocabulary(word_counts)
        sub_words = miss_text.learn_word_piece(
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
            adapts = np.char.decode(adapts.numpy().ravel().astype('S'), 'utf-8')
            for adapt in adapts:
                adapt_counts[adapt] += word_counts[word]

        return adapt_counts

    def adapt(self, inputs):
        self.lookup()  # Init lookup table

        adapts = super().adapt(inputs)
        subwords = miss_text.word_piece(
            source=adapts,
            lookup_table=self._lookup.lookup_table,
            joiner_prefix=self.joiner_prefix,
            max_bytes=(self.max_len or 9999) * 4,
            max_chars=0,
            unknown_token=self.UNK_MARK,
            split_unknown=True,
            skip=self._reserved_words)

        return subwords

    @shape_type_conversion
    def build(self, input_shape=None):
        self.reduce = Reduction(self.reduction)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not self.with_prep and not isinstance(inputs, tf.RaggedTensor):
            raise ValueError('Expecting `inputs` to be `RaggedTensor` instance.')

        embeds = super().call(inputs)
        outputs = self.reduce(embeds)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        prep_shape = input_shape
        if not self.with_prep:
            prep_shape = input_shape[:-1]

        return prep_shape + (self.output_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'upper_thresh': self.upper_thresh,
            'lower_thresh': self.lower_thresh,
            'num_iterations': self.num_iterations,
            'max_tokens': self.max_tokens,
            'max_chars': self.max_chars,
            'slack_ratio': self.slack_ratio,
            'include_joiner': self.include_joiner,
            'joiner_prefix': self.joiner_prefix,
            'reduction': self.reduction
        })

        return config


@register_keras_serializable(package='Miss')
class CnnEmbedding(WordEmbedding):
    BOW_MARK = '[BOW]'
    EOW_MARK = '[EOW]'

    def __init__(self, vocabulary=None, output_dim=None, kernels=(1, 2, 3, 4, 5, 6, 7),
                 filters=(32, 32, 64, 128, 256, 512, 1024), char_dim=16, activation='tanh', highways=2,
                 normalize_unicode='NFKC', lower_case=False, zero_digits=False, max_len=None, reserved_words=None,
                 embed_type='dense_auto', adapt_cutoff=None, adapt_factor=4, with_prep=False,
                 embeddings_initializer=initializers.random_uniform(-1., 1.), **kwargs):

        _reserved_words = [self.BOW_MARK, self.EOW_MARK]
        _reserved_words += [] if reserved_words is None else [r for r in reserved_words if r not in _reserved_words]

        super().__init__(vocabulary=vocabulary, output_dim=output_dim, normalize_unicode=normalize_unicode,
                         lower_case=lower_case, zero_digits=zero_digits, max_len=max_len,
                         reserved_words=_reserved_words, embed_type=embed_type, adapt_cutoff=adapt_cutoff,
                         adapt_factor=adapt_factor, with_prep=with_prep, embeddings_initializer=embeddings_initializer,
                         **kwargs)

        if not kernels or not isinstance(kernels, (list, tuple)) or \
                not all(map(lambda x: isinstance(x, int), kernels)):
            raise ValueError('Expecting "kernels" to be a list of integers')
        if not filters or not isinstance(filters, (list, tuple)) or \
                not all(map(lambda x: isinstance(x, int), filters)):
            raise ValueError('Expecting "filters" to be a list of integers')
        if len(kernels) != len(filters):
            raise ValueError('Sizes of "kernels" and "filters" should be equal')
        self.kernels = kernels
        self.filters = filters
        self.char_dim = char_dim
        self.activation = activations.get(activation)
        self.highways = highways

    def adapt(self, inputs):
        adapts = super().adapt(inputs)
        chars = miss_text.split_chars(adapts, skip=self._reserved_words)

        values = adapts
        if isinstance(adapts, tf.RaggedTensor):
            values = adapts.flat_values
        shape = tf.shape(values)

        bos = tf.fill(shape, self.BOW_MARK)
        eos = tf.fill(shape, self.EOW_MARK)

        if isinstance(inputs, tf.RaggedTensor):
            bos = adapts.with_flat_values(bos)
            eos = adapts.with_flat_values(eos)

        wraps = tf.concat([bos[..., None], chars, eos[..., None]], axis=-1)

        return wraps

    def build(self, input_shape=None):
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

    def call(self, inputs, **kwargs):
        if not self.with_prep and not isinstance(inputs, tf.RaggedTensor):
            raise ValueError('Expecting `inputs` to be `RaggedTensor` instance.')

        if self.with_prep:
            inputs = self.preprocess(inputs)

        rows = inputs.nested_row_lengths()
        flats = tf.RaggedTensor.from_row_lengths(inputs.flat_values, rows[-1])

        # embeds = super().call(flats) # without preprocessing
        embeds = self.embed(flats)
        embeds = embeds.to_tensor()

        convs = [c(embeds) for c in self.conv]
        convs = [self.pool(c) for c in convs]
        convs = [self.act(c) for c in convs]
        convs = tf.concat(convs, axis=-1)

        outputs = convs
        for h in self.high:
            outputs = h(outputs)

        if self.proj is not None:
            outputs = self.proj(outputs)

        outputs = tf.RaggedTensor.from_nested_row_lengths(outputs, rows[:-1])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        prep_shape = input_shape
        if not self.with_prep:
            prep_shape = input_shape[:-1]

        return prep_shape + (self.output_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernels': self.kernels,
            'filters': self.filters,
            'char_dim': self.char_dim,
            'activation': activations.serialize(self.activation),
            'highways': self.highways
        })

        return config


@register_keras_serializable(package='Miss')
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

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
