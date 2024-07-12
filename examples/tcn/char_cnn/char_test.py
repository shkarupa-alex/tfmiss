import argparse
import numpy as np
from matplotlib import pyplot
from tf_keras import optimizers
from model import CharModel
from utils import data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout applied to layers')
    parser.add_argument('--embed_dropout', type=float, default=0.1, help='Dropout applied to the embedded layer')
    parser.add_argument('--clip', type=float, default=0.15, help='Gradient clip')
    parser.add_argument('--epochs', type=int, default=100, help='Upper epoch limit')
    parser.add_argument('--ksize', type=int, default=3, help='Kernel size')
    parser.add_argument('--levels', type=int, default=3, help='Number of levels')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--embed_size', type=int, default=100, help='Dimension of character embeddings')
    parser.add_argument('--nhid', type=int, default=450, help='Number of hidden units per layer')
    parser.add_argument('--validseqlen', type=int, default=320, help='Valid sequence length')
    parser.add_argument('--seq_len', type=int, default=400, help='Total sequence length, including effective history')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--dataset', type=str, default='ptb', help='Dataset to use')
    parser.add_argument('--gru_units', type=int, default=1000, help='Number of hidden units for GRU layer')
    parser.add_argument('--lstm_units', type=int, default=850, help='Number of hidden units for LSTM layer')
    argv, _ = parser.parse_known_args()

    np.random.seed(argv.seed)

    train_dataset, test_dataset, vocab_size = data_generator(argv.dataset, argv.seq_len, argv.batch_size)

    filters = {
        CharModel.CORE_GRU: [argv.gru_units],
        CharModel.CORE_LSTM: [argv.lstm_units],
        CharModel.CORE_TCN: [argv.nhid] * argv.levels,
        CharModel.CORE_TCN_HE: [argv.nhid] * argv.levels,
    }

    histories = {}
    for core in [CharModel.CORE_LSTM, CharModel.CORE_GRU, CharModel.CORE_TCN, CharModel.CORE_TCN_HE]:
        model = CharModel(
            seq_len=argv.seq_len,
            vocab_size=vocab_size,
            embed_size=argv.embed_size,
            core=core,
            filters=filters[core],
            kernel_size=argv.ksize,
            dropout=argv.dropout,
            embed_dropout=argv.embed_dropout
        )
        model.compile(
            run_eagerly=False,
            optimizer=optimizers.Adam(argv.lr, clipnorm=argv.clip),
            loss='sparse_categorical_crossentropy',
        )
        model.summary()
        histories[core] = model.fit(
            train_dataset,
            epochs=argv.epochs,
            validation_data=test_dataset,
        ).history

    interval = np.linspace(0, argv.epochs, argv.epochs)
    cmap = pyplot.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(histories.items())))

    for (core, m), color in zip(histories.items(), colors):
        pyplot.plot(interval, m['loss'], color=color, label=core)

    pyplot.title('Character LM on {}'.format(argv.dataset.upper()))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('loss_{}.png'.format(argv.dataset))
    pyplot.close()
