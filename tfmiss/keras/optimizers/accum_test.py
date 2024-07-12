import tensorflow as tf
from absl.testing import parameterized
from keras import optimizers
from keras.optimizers.schedules import PiecewiseConstantDecay
from keras.src.testing_infra import test_combinations
from tfmiss.keras.optimizers.accum import Accum


class AccumOptimizerTest(test_combinations.TestCase):
    @parameterized.parameters([
        ('SGD', 0.01, 0.01, None), ('SGD', 0.01, 0.01, 0.05), ('Adam', 0.01, 0.01, None), ('Adam', 0.01, 0.01, 0.005),
        ('Adam', PiecewiseConstantDecay([9, 19], [0.01, 0.02, 0.03]),
         PiecewiseConstantDecay([29, 59], [0.01, 0.02, 0.03]), None)])
    def test_dense_value(self, opt, lr3, lr1, wd):
        logits = tf.constant([
            [[0.3834889522317029, 0.18652422068782437], [0.5416385105321877, 0.7688347307618931],
             [0.217790919325616, 0.15262361602508145], [0.2878849762271264, 0.303517183413756]],
            [[0.575396293440796, 0.1937945842042471], [0.8515048936764875, 0.6082759528597521],
             [0.244061034387261, 0.6003517264187057], [0.6859539966321943, 0.5198978593981205]],
            [[0.3610914004327209, 0.36862621854042843], [0.5987028639328222, 0.659444080108686],
             [0.007242132828604975, 0.8057883687901827], [0.2878444083304914, 0.8037585089042157]],
            [[0.9682742254823508, 0.8033685400744861], [0.5634681959664141, 0.866969703248222],
             [0.15559689502741003, 0.2019452510527322], [0.5022437034539109, 0.02060094585527439]],
            [[0.07024793255130724, 0.9250163595307129], [0.7577723801679607, 0.9831952010937728],
             [0.29071966476410227, 0.6668720294403833], [0.5692100239777321, 0.0768018007818857]],
            [[0.9364803726429763, 0.003048170334680189], [0.2174190664238198, 0.5639216930749434],
             [0.03810156819314736, 0.9079015544559597], [0.45384361800440465, 0.23852113113902818]],
            [[0.7612423837745981, 0.07139997292488698], [0.7691063629177809, 0.4477965784803455],
             [0.45369956264823375, 0.610529191717423], [0.9398398146291436, 0.8149969007287517]],
            [[0.8523592275802878, 0.5669437663867238], [0.3755580315463406, 0.5345157509149302],
             [0.49289132309138584, 0.2624251305383454], [0.8983067036877381, 0.29240395916025474]],
            [[0.8249801893607751, 0.796818088911959], [0.8452020180686657, 0.12773879285629697],
             [0.5129248509630417, 0.6258146179509537], [0.7802920713460937, 0.21178474682109627]]], 'float32')
        targets = tf.constant([
            [[0.3706484442126573], [0.9120088220681468], [0.8002354619370015], [0.1404340849024026]],
            [[0.6983717383122452], [0.7516800505500254], [0.8739545056835875], [0.4384932111106432]],
            [[0.6527435852360429], [0.7623048790390828], [0.38295686895742076], [0.045712707742929126]],
            [[0.6818675631406735], [0.6233808405487737], [0.9570485572286584], [0.03659283629384047]],
            [[0.3349266363012273], [0.17621407995129712], [0.06932226643333994], [0.2133203445158769]],
            [[0.4572859905619785], [0.08999689428717572], [0.7617251196788672], [0.46156182410210855]],
            [[0.6886897863452727], [0.8836154248090233], [0.4212059466588135], [0.10048563036423053]],
            [[0.916738019979103], [0.48745919502299206], [0.3728910185868768], [0.1744337592170927]],
            [[0.8880334776379646], [0.6088421798176064], [0.9071319598757348], [0.043805577709025156]]], 'float32')
        initial = [[0.9189467373291287], [0.5046289624616127]]

        weights = tf.Variable(initial, trainable=True, dtype='float32')
        optimizer = optimizers.get({'class_name': opt, 'config': {
            'learning_rate': lr3, 'weight_decay': wd, 'is_legacy_optimizer': False}})
        logits_, targets_ = tf.reshape(logits, [3, 12, 2]), tf.reshape(targets, [3, 12, 1])

        expected = []
        for e in range(10):
            for b in range(3):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean((targets_[b] - logits_[b] @ weights) ** 2)
                grads = tape.gradient(loss, [weights])
                optimizer.apply_gradients(zip(grads, [weights]))

                actual = self.evaluate(weights)
                expected.append(actual)

        weights = tf.Variable(initial, trainable=True, dtype='float32')
        optimizer = Accum(optimizers.get(
            {'class_name': opt, 'config': {
                'learning_rate': lr1, 'weight_decay': wd, 'is_legacy_optimizer': False}}), 3)

        for e in range(10):
            for b in range(9):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean((targets[b] - logits[b] @ weights) ** 2)
                grads = tape.gradient(loss, [weights])
                optimizer.apply_gradients(zip(grads, [weights]))

                if 2 == b % 3:
                    actual = self.evaluate(weights)
                    self.assertAllClose(actual, expected[(e * 9 + b - 1) // 3])

    def test_without_sparse(self):
        logits = tf.random.uniform((10, 4, 2), dtype='float32')
        targets = tf.random.uniform((10, 4, 1), dtype='float32')

        weights = tf.Variable([[0.9189467373291287], [0.5046289624616127]], trainable=True, dtype='float32')
        optimizer = Accum('Adam', 3, sparse_support=False)

        for e in range(10):
            for b in range(9):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean((targets[b] - logits[b] @ weights) ** 2)
                grads = tape.gradient(loss, [weights])
                optimizer.apply_gradients(zip(grads, [weights]))

        actual = self.evaluate(weights).tolist()
        self.assertAllClose(actual, [[0.8901770710945129], [0.4755144417285919]])

    @parameterized.parameters([
        ('SGD', 0.01, 0.01), ('Adam', 0.01, 0.01),
        ('Adam', PiecewiseConstantDecay([9, 19], [0.01, 0.02, 0.03]),
         PiecewiseConstantDecay([29, 59], [0.01, 0.02, 0.03]))])
    def test_sparse_value(self, opt, lr3, lr1):
        indices = tf.constant([
            [0, 1, 2, 3], [4, 3, 2, 1], [0, 1, 2, 3], [4, 3, 2, 1], [0, 1, 2, 3], [4, 3, 2, 1], [0, 0, 1, 1],
            [2, 2, 2, 3], [3, 3, 3, 3]], 'int32')
        targets = tf.ones([9, 4, 2], 'float32')
        initial = [
            [0.23922160713307927, 0.3744851446884956], [0.12467899195548227, 0.582473576084052],
            [0.03158169648583875, 0.5722574065524645], [0.8287652988660922, 0.315956053134146],
            [0.09411076902398452, 0.33044468551739103]]

        weights = tf.Variable(initial, trainable=True, dtype='float32')
        optimizer = optimizers.get({'class_name': opt, 'config': {'learning_rate': lr3, 'is_legacy_optimizer': False}})
        indices_, targets_ = tf.reshape(indices, [3, 12]), tf.reshape(targets, [3, 12, 2])

        expected = []
        for e in range(10):
            for b in range(3):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean((targets_[b] - tf.nn.embedding_lookup(weights, indices_[b])) ** 2)
                grads = tape.gradient(loss, [weights])
                optimizer.apply_gradients(zip(grads, [weights]))

                actual = self.evaluate(weights)
                expected.append(actual)

        weights = tf.Variable(initial, trainable=True, dtype='float32')
        optimizer = Accum(optimizers.get(
            {'class_name': opt, 'config': {'learning_rate': lr1, 'is_legacy_optimizer': False}}), 3)

        for e in range(10):
            for b in range(9):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean((targets[b] - tf.nn.embedding_lookup(weights, indices[b])) ** 2)
                grads = tape.gradient(loss, [weights])
                optimizer.apply_gradients(zip(grads, [weights]))

                if 2 == b % 3:
                    actual = self.evaluate(weights)
                    self.assertAllClose(actual, expected[(e * 9 + b - 1) // 3], atol=1e-5)

    def test_wrapper(self):
        optimizer = optimizers.Adam(0.15)
        wrapped = Accum(optimizer, 3)

        self.assertEqual(optimizer.learning_rate, wrapped.learning_rate)
        self.assertEqual(optimizer.beta_1, wrapped.beta_1)

    def test_lr(self):
        opt_1 = Accum(optimizers.Adam(learning_rate=1.0), 4)
        opt_2 = Accum(optimizers.Adam(learning_rate=lambda: tf.constant(0.5, 'float32')), 4)
        opt_3 = Accum.from_config(opt_2.get_config())

        self.assertIsInstance(opt_1.lr, tf.Variable)
        self.assertIsInstance(opt_2.lr, tf.Variable)
        self.assertIsInstance(opt_3.lr, tf.Variable)

        self.assertAllClose(self.evaluate(opt_1.learning_rate), 1.0)
        self.assertAllClose(self.evaluate(opt_2.learning_rate), 0.5)
        self.assertAllClose(self.evaluate(opt_3.learning_rate), 0.5)


if __name__ == "__main__":
    tf.test.main()
