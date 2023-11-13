from unittest import TestCase, main
from class_HMM import HMM


class HMMTest(TestCase):
    def test_init(self):
        hmm_obj = HMM(10, 10)
        self.assertEqual(hmm_obj.X, 10)
        self.assertEqual(hmm_obj.Y, 10)
        self.assertEqual(hmm_obj.matrix, [])
        self.assertEqual(hmm_obj.shuffled, [])
        self.assertEqual(hmm_obj.Z2, [])

    def test_concat_SV(self):
        X = [1, 2]
        Y = [3, 4]
        hmm_obj = HMM(X, Y)
        self.assertEqual(hmm_obj.concat_SV(X, Y), [[1], [2], [3], [4]])

    def test_shuffle_matrix(self):
        X = [1, 2]
        Y = [3, 4, 5]
        hmm_obj = HMM(X, Y)
        self.assertNotEqual(hmm_obj.shuffle_matrix(), [[1], [2], [3], [4], [5]])

    def test_Gauss_model(self):
        X = [1, 2, 3, 4]
        Y = [10, 15, 41, 19]
        hmm_obj = HMM(X, Y)
        hmm_obj.shuffle_matrix()
        self.assertEqual(len(hmm_obj.Gaussian_model()), 8)

    def test_GMMHMM_model(self):
        X = [1, 2, 3]
        Y = [10, 15, 41, 19]
        hmm_obj = HMM(X, Y)
        hmm_obj.shuffle_matrix()
        self.assertEqual(len(hmm_obj.GMMHMM_model()), 7)

    def test_check_true(self):
        X = [11.67832142164908, 9.971898675188816, 25.01841276406762, 15.03967147266212]
        Y = [111.73482137940924, 108.3107229866111, 88.27019604202627, 120.30844003754476]
        hmm_obj = HMM(X, Y)
        hmm_obj.shuffle_matrix()
        hmm_obj.Gaussian_model()
        self.assertEqual(hmm_obj.check_correct_predict(), True)

    def test_check_false(self):
        X = [1, 0]
        Y = [35.37551597648314, 52.25567792055704, 55.671974648818214, 0.1131892096802, 0.1131892096802]
        hmm_obj = HMM(X, Y)
        hmm_obj.shuffle_matrix()
        hmm_obj.Gaussian_model()
        self.assertEqual(hmm_obj.check_correct_predict(), False)


if __name__ == '__main__':
    main()