from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.training.bucket import init_buckets, waste_frac, merge_buckets, merge_allowed, group_buckets
from tfmiss.training.bucket import estimate_bucket_boundaries, estimate_bucket_pipeline


class InitBucketsTest(tf.test.TestCase):
    def setUp(self):
        super(InitBucketsTest, self).setUp()
        self.len2freq = {
            255: 16, 256: 15, 257: 20, 258: 16, 259: 17, 260: 15, 261: 15, 262: 12, 263: 13, 264: 13, 265: 11, 266: 9,
            267: 8, 268: 9, 269: 7, 270: 9, 271: 7, 272: 6, 273: 5, 274: 6, 275: 5, 276: 4, 277: 4, 278: 4, 279: 4,
            280: 4, 281: 5, 282: 3, 283: 3, 284: 3, 285: 3, 286: 2, 287: 3, 288: 2, 289: 2, 290: 3, 291: 2, 292: 1,
            293: 2, 294: 1, 295: 2, 296: 1, 297: 1, 298: 1, 300: 1, 301: 1, 303: 1, 304: 1, 305: 1, 311: 1
        }

    def test_error(self):
        with self.assertRaisesRegexp(ValueError, 'Empty'):
            init_buckets({}, mod8=False)

        with self.assertRaisesRegexp(ValueError, 'Keys'):
            init_buckets({'1': 2}, mod8=False)

        with self.assertRaisesRegexp(ValueError, 'Values'):
            init_buckets({1: 2.0}, mod8=False)

    def test_mod_1(self):
        expected = [
            (256, {255: 16}), (257, {256: 15}), (258, {257: 20}), (259, {258: 16}), (260, {259: 17}), (261, {260: 15}),
            (262, {261: 15}), (263, {262: 12}), (264, {263: 13}), (265, {264: 13}), (266, {265: 11}), (267, {266: 9}),
            (268, {267: 8}), (269, {268: 9}), (270, {269: 7}), (271, {270: 9}), (272, {271: 7}), (273, {272: 6}),
            (274, {273: 5}), (275, {274: 6}), (276, {275: 5}), (277, {276: 4}), (278, {277: 4}), (279, {278: 4}),
            (280, {279: 4}), (281, {280: 4}), (282, {281: 5}), (283, {282: 3}), (284, {283: 3}), (285, {284: 3}),
            (286, {285: 3}), (287, {286: 2}), (288, {287: 3}), (289, {288: 2}), (290, {289: 2}), (291, {290: 3}),
            (292, {291: 2}), (293, {292: 1}), (294, {293: 2}), (295, {294: 1}), (296, {295: 2}), (297, {296: 1}),
            (298, {297: 1}), (299, {298: 1}), (301, {300: 1}), (302, {301: 1}), (304, {303: 1}), (305, {304: 1}),
            (306, {305: 1}), (312, {311: 1})
        ]
        buckets = init_buckets(self.len2freq, mod8=False)
        self.assertListEqual(expected, buckets)

    def test_mod_8(self):
        expected = [
            (257, {256: 15, 255: 16}), (265, {257: 20, 258: 16, 259: 17, 260: 15, 261: 15, 262: 12, 263: 13, 264: 13}),
            (273, {265: 11, 266: 9, 267: 8, 268: 9, 269: 7, 270: 9, 271: 7, 272: 6}),
            (281, {273: 5, 274: 6, 275: 5, 276: 4, 277: 4, 278: 4, 279: 4, 280: 4}),
            (289, {288: 2, 281: 5, 282: 3, 283: 3, 284: 3, 285: 3, 286: 2, 287: 3}),
            (297, {289: 2, 290: 3, 291: 2, 292: 1, 293: 2, 294: 1, 295: 2, 296: 1}),
            (305, {297: 1, 298: 1, 300: 1, 301: 1, 303: 1, 304: 1}), (313, {305: 1, 311: 1})
        ]
        buckets = init_buckets(self.len2freq, mod8=True)
        self.assertListEqual(expected, buckets)


class WasteFracTest(tf.test.TestCase):
    def test_empty(self):
        result = waste_frac(tuple())
        self.assertEqual(0.0, result)

    def test_no_waste(self):
        result = waste_frac((4, {3: 5}))
        self.assertEqual(0.0, result)

    def test_normal(self):
        result = waste_frac((4, {1: 5, 2: 2, 3: 1}))
        self.assertEqual(0.5, result)


class MergeBucketsTest(tf.test.TestCase):
    def test_empty(self):
        result = merge_buckets(tuple(), tuple())
        self.assertEqual(tuple(), result)

        source = (4, {3: 5})
        result = merge_buckets(tuple(), source)
        self.assertEqual(source, result)

        result = merge_buckets(source, tuple())
        self.assertEqual(source, result)

    def test_normal(self):
        result = merge_buckets((4, {3: 5}), (7, {6: 4, 5: 3}))
        self.assertEqual((7, {6: 4, 5: 3, 3: 5}), result)


class MergeAllowedTest(tf.test.TestCase):
    def setUp(self):
        super(MergeAllowedTest, self).setUp()
        self.buckets = [
            (257, {256: 15, 255: 16}), (265, {257: 20, 258: 16, 259: 17, 260: 15, 261: 15, 262: 12, 263: 13, 264: 13}),
            (273, {265: 11, 266: 9, 267: 8, 268: 9, 269: 7, 270: 9, 271: 7, 272: 6}),
            (281, {273: 5, 274: 6, 275: 5, 276: 4, 277: 4, 278: 4, 279: 4, 280: 4}),
            (289, {288: 2, 281: 5, 282: 3, 283: 3, 284: 3, 285: 3, 286: 2, 287: 3}),
            (297, {289: 2, 290: 3, 291: 2, 292: 1, 293: 2, 294: 1, 295: 2, 296: 1}),
            (305, {297: 1, 298: 1, 300: 1, 301: 1, 303: 1, 304: 1}), (313, {305: 1, 311: 1})
        ]

    def test_empty(self):
        result = merge_allowed(tuple(), self.buckets, min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertEqual(False, result)

    def test_low_waste(self):
        # wasted 0.002
        # aggregated 0.103
        result = merge_allowed(self.buckets[0], self.buckets, min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertEqual(True, result)

    def test_high_waste_low_aggr(self):
        # wasted 0.013
        # aggregated 0.007
        result = merge_allowed(self.buckets[-1], self.buckets, min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertEqual(True, result)

    def test_high_waste_high_aggr(self):
        # wasted 0.014
        # aggregated 0.220
        result = merge_allowed(self.buckets[2], self.buckets, min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertEqual(False, result)


class GroupBucketsTest(tf.test.TestCase):
    def setUp(self):
        super(GroupBucketsTest, self).setUp()
        self.buckets = [
            (256, {255: 16}), (257, {256: 15}), (258, {257: 20}), (259, {258: 16}), (260, {259: 17}), (261, {260: 15}),
            (262, {261: 15}), (263, {262: 12}), (264, {263: 13}), (265, {264: 13}), (266, {265: 11}), (267, {266: 9}),
            (268, {267: 8}), (269, {268: 9}), (270, {269: 7}), (271, {270: 9}), (272, {271: 7}), (273, {272: 6}),
            (274, {273: 5}), (275, {274: 6}), (276, {275: 5}), (277, {276: 4}), (278, {277: 4}), (279, {278: 4}),
            (280, {279: 4}), (281, {280: 4}), (282, {281: 5}), (283, {282: 3}), (284, {283: 3}), (285, {284: 3}),
            (286, {285: 3}), (287, {286: 2}), (288, {287: 3}), (289, {288: 2}), (290, {289: 2}), (291, {290: 3}),
            (292, {291: 2}), (293, {292: 1}), (294, {293: 2}), (295, {294: 1}), (296, {295: 2}), (297, {296: 1}),
            (298, {297: 1}), (299, {298: 1}), (301, {300: 1}), (302, {301: 1}), (304, {303: 1}), (305, {304: 1}),
            (306, {305: 1}), (312, {311: 1})
        ]

    def test_empty(self):
        result = group_buckets([], self.buckets[0], [], min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertEqual(3, len(result))
        self.assertListEqual([], result[0])
        self.assertListEqual([self.buckets[0]], [result[1]])
        self.assertListEqual([], result[2])

    def test_group(self):
        expected = (258, {256: 15, 257: 20, 255: 16})
        result = group_buckets(self.buckets[:1], self.buckets[1], self.buckets[2:],
                                min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertListEqual([expected], [result[1]])

    def test_no_group(self):
        expected = (312, {311: 1})
        result = group_buckets(self.buckets[:-1], self.buckets[-1], [], min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertListEqual([expected], [result[1]])


class EstimateBucketBoundariesTest(tf.test.TestCase):
    def setUp(self):
        super(EstimateBucketBoundariesTest, self).setUp()
        self.len2freq = {
            255: 16, 256: 15, 257: 20, 258: 16, 259: 17, 260: 15, 261: 15, 262: 12, 263: 13, 264: 13, 265: 11, 266: 9,
            267: 8, 268: 9, 269: 7, 270: 9, 271: 7, 272: 6, 273: 5, 274: 6, 275: 5, 276: 4, 277: 4, 278: 4, 279: 4,
            280: 4, 281: 5, 282: 3, 283: 3, 284: 3, 285: 3, 286: 2, 287: 3, 288: 2, 289: 2, 290: 3, 291: 2, 292: 1,
            293: 2, 294: 1, 295: 2, 296: 1, 297: 1, 298: 1, 300: 1, 301: 1, 303: 1, 304: 1, 305: 1, 311: 1
        }

    def test_mod_1(self):
        expected = [258, 263, 268, 273, 279, 285, 291, 297, 302, 306, 312]
        result = estimate_bucket_boundaries(self.len2freq, mod8=False, min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertListEqual(expected, result)
        self.assertLess(len(result), len(self.len2freq.keys()))

    def test_mod_8(self):
        expected = [257, 265, 273, 281, 289, 297, 305, 313]
        result = estimate_bucket_boundaries(self.len2freq, mod8=True, min_waste=0.01, max_waste=0.1, min_aggr=0.01)
        self.assertListEqual(expected, result)
        self.assertLess(len(result), len(self.len2freq.keys()))


class EstimateBucketPipelineTest(tf.test.TestCase):
    def test_unsafe_mod_1(self):
        source_buckets = [262, 268, 274, 281, 287, 294, 301]
        num_samples = 100000
        expected_batches = [383, 375, 366, 357, 350, 341, 333]

        buckets, batches, max_bound = estimate_bucket_pipeline(
            bucket_boundaries=source_buckets,
            num_samples=num_samples, safe=False, mod8=False)

        self.assertEqual(len(buckets) + 1, len(batches))
        self.assertListEqual(source_buckets[:-1], buckets)
        self.assertListEqual(expected_batches, batches)
        self.assertEqual(source_buckets[-1], max_bound)

        samples_delta = [(num_samples - batch * (bucket - 1)) / num_samples
                         for bucket, batch in zip(buckets + [max_bound], batches)]
        self.assertAllLess(samples_delta, 0.002)

    def test_safe_mod_1(self):
        source_buckets = [262, 268, 274, 281, 287, 294, 301]
        num_samples = 100000
        expected_batches = [383, 374, 366, 357, 349, 341, 333]

        buckets, batches, max_bound = estimate_bucket_pipeline(
            bucket_boundaries=source_buckets,
            num_samples=num_samples, safe=True, mod8=False)

        self.assertEqual(len(buckets) + 1, len(batches))
        self.assertListEqual(source_buckets[:-1], buckets)
        self.assertListEqual(expected_batches, batches)
        self.assertEqual(source_buckets[-1], max_bound)

        samples_delta = [(num_samples - batch * (bucket - 1)) / num_samples
                         for bucket, batch in zip(buckets + [max_bound], batches)]
        self.assertAllLess(samples_delta, 0.002)
        self.assertAllGreater(samples_delta, 0.0)

    def test_unsafe_mod_8(self):
        source_buckets = [262, 268, 274, 281, 287, 294, 301]
        num_samples = 100000
        expected_batches = [384, 376, 368, 360, 352, 344, 336]

        buckets, batches, max_bound = estimate_bucket_pipeline(
            bucket_boundaries=source_buckets,
            num_samples=num_samples, safe=False, mod8=True)

        self.assertEqual(len(buckets) + 1, len(batches))
        self.assertListEqual(source_buckets[:-1], buckets)
        self.assertListEqual(expected_batches, batches)
        self.assertAllEqual([0] * len(batches), [b % 8 for b in batches])
        self.assertEqual(source_buckets[-1], max_bound)

        samples_delta = [(num_samples - batch * (bucket - 1)) / num_samples
                         for bucket, batch in zip(buckets + [max_bound], batches)]
        self.assertAllLess(samples_delta, 0.02)

    def test_safe_mod_8(self):
        source_buckets = [262, 268, 274, 281, 287, 294, 301]
        num_samples = 100000
        expected_batches = [376, 368, 360, 352, 344, 336, 328]

        buckets, batches, max_bound = estimate_bucket_pipeline(
            bucket_boundaries=source_buckets,
            num_samples=num_samples, safe=True, mod8=True)

        self.assertEqual(len(buckets) + 1, len(batches))
        self.assertListEqual(source_buckets[:-1], buckets)
        self.assertListEqual(expected_batches, batches)
        self.assertAllEqual([0] * len(batches), [b % 8 for b in batches])
        self.assertEqual(source_buckets[-1], max_bound)

        samples_delta = [(num_samples - batch * (bucket - 1)) / num_samples
                         for bucket, batch in zip(buckets + [max_bound], batches)]
        self.assertAllLess(samples_delta, 0.02)
        self.assertAllGreater(samples_delta, 0.0)

    def test_safe_mod8_trim(self):
        source_buckets = [65, 129, 257, 513, 1025, 2049, 4097]
        num_samples = 10000
        expected_buckets = [65, 129, 257, 513]
        expected_batches = [152, 72, 32, 16, 8]
        expected_max = 1025

        buckets, batches, max_bound = estimate_bucket_pipeline(
            bucket_boundaries=source_buckets,
            num_samples=num_samples, safe=True, mod8=True)

        self.assertEqual(len(buckets) + 1, len(batches))
        self.assertListEqual(expected_buckets, buckets)
        self.assertListEqual(expected_batches, batches)
        self.assertAllEqual([0] * len(batches), [b % 8 for b in batches])
        self.assertEqual(expected_max, max_bound)

        samples_delta = [(num_samples - batch * (bucket - 1)) / num_samples
                         for bucket, batch in zip(buckets + [max_bound], batches)]
        self.assertAllLess(samples_delta, 0.2)
        self.assertAllGreater(samples_delta, 0.0)

    def test_safe_mod8_trim_error(self):
        source_buckets = [65, 129, 257, 513, 1025, 2049, 4097]
        num_samples = 1000
        with self.assertRaisesRegexp(ValueError, 'few samples per batch'):
            estimate_bucket_pipeline(
                bucket_boundaries=source_buckets,
                num_samples=num_samples, safe=True, mod8=True)


if __name__ == '__main__':
    tf.test.main()
