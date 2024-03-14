import argparse
import numpy as np
import time
from matplotlib import pyplot
from keras import optimizers
from model import Text8Model
from utils import data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout applied to layers')
    parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--embed_size', type=int, default=100, help='Dimension of character embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='Number of hidden units per layer')
    parser.add_argument('--seq_len', type=int, default=20, help='Total sequence length, including effective history')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--cutoff', type=str, default='2000,10000', help='Adaptive softmax cutoffs')
    parser.add_argument('--negatives', type=int, default=10, help='Number of negative examples for sampled softmaxes')
    parser.add_argument('--return_probs', action='store_true', default=False)
    argv, _ = parser.parse_known_args()

    argv_cutoff = [int(cf) for cf in argv.cutoff.split(',')]
    if len(argv_cutoff) < 2:
        raise ValueError('Wrong adaptive softmax cutoff specified')

    np.random.seed(argv.seed)

    train_dataset, test_dataset, vocab_size = data_generator(argv.seq_len, argv.batch_size)

    titles = ['Softmax', 'SampledSoftmax', 'AdaptiveSoftmax (default)',
              'AdaptiveSoftmax (accurate)', 'AdaptiveSoftmax (fast)']
    models = [
        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_SM,
            dropout=argv.dropout,
            cutoff=argv_cutoff,
            negatives=argv.negatives,
            return_probs=argv.return_probs
        ),

        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_SS,
            dropout=argv.dropout,
            cutoff=argv_cutoff,
            negatives=argv.negatives,
            return_probs=argv.return_probs
        ),

        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_ASM,
            dropout=argv.dropout,
            cutoff=argv_cutoff,
            negatives=argv.negatives,
            return_probs=argv.return_probs
        ),
        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_ASM,
            dropout=argv.dropout,
            cutoff=[3526, 13950],
            negatives=argv.negatives,
            return_probs=argv.return_probs
        ),
        Text8Model(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            units=argv.nhid,
            core=Text8Model.OUT_ASM,
            dropout=argv.dropout,
            cutoff=[454, 7358],
            negatives=argv.negatives,
            return_probs=argv.return_probs
        ),
    ]

    histories = {}
    for model, title in zip(models, titles):
        loss = 'sparse_categorical_crossentropy' if 'Softmax' == title else None
        model.compile(
            run_eagerly=False,
            optimizer=optimizers.SGD(argv.lr),
            loss=loss,
        )
        model.summary()

        start_time = time.time()
        histories[title] = model.fit(
            train_dataset,
            epochs=argv.epochs,
            validation_data=test_dataset,
        ).history
        histories[title]['time'] = (time.time() - start_time) / 60
        histories[title]['ce'] = histories[title]['val_loss']

    cmap = pyplot.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(histories.items())))

    for (core, m), color in zip(histories.items(), colors):
        interval = np.linspace(0, m['time'], argv.epochs)
        pyplot.plot(interval, m['ce'], color=color, label=core)

    pyplot.title('Text8 word LM')
    pyplot.xlabel('Time (minutes)')
    pyplot.ylabel('Crossentropy')
    pyplot.legend()
    if argv.return_probs:
        pyplot.savefig('loss.png')
    else:
        pyplot.savefig('speed_no_probs.png')
    pyplot.close()
