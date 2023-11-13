from unittest import TestCase, main
from class_SV import SV, ExpSV, NormSV
from class_HMM import HMM


class SVTest(TestCase):
    def test_init(self):
        sv_obj = SV(10)
        self.assertEqual(sv_obj.count, 10)
        self.assertEqual(sv_obj.numbers, [])

    def test_generate_base_numbers(self):
        sv_obj = SV(12)
        sv_obj._generate_random_base_numbers()
        self.assertEqual(len(sv_obj.numbers), 12)


class ExpTest(TestCase):
    def test_init(self):
        exp_obj = ExpSV(10, 0.1)
        self.assertEqual(exp_obj.count, 10)
        self.assertEqual(exp_obj.Lambda, 0.1)
        self.assertEqual(exp_obj.rasp, [])

    def test_generate_exp(self):
        exp_obj = ExpSV(10, 0.1)
        exp_obj.generate_exp_rasp()
        self.assertEqual(len(exp_obj.rasp), 10)


class NormTest(TestCase):
    def test_init(self):
        norm_obj = NormSV(10, 100, 30)
        self.assertEqual(norm_obj.count, 10)
        self.assertEqual(norm_obj.Tsp, 100)
        self.assertEqual(norm_obj.Sigma, 30)
        self.assertEqual(norm_obj.rasp, [])

    def test_generate_rasp(self):
        norm_obj = NormSV(10, 100, 30)
        norm_obj.generate_norm_rasp()
        self.assertEqual(len(norm_obj.rasp), 10)


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


class IntegrationTest(TestCase):
    def test_use_two_exp_rasps_in_hmm(self):
        exp_obj1 = ExpSV(3, 0.1)
        exp_obj2 = ExpSV(4, 0.3)
        exp_obj1.generate_exp_rasp()
        exp_obj2.generate_exp_rasp()
        hmm_obj = HMM(exp_obj1.rasp, exp_obj2.rasp)
        self.assertEqual(hmm_obj.X, exp_obj1.rasp)
        self.assertEqual(hmm_obj.Y, exp_obj2.rasp)

    def test_use_two_norm_rasps_in_hmm(self):
        norm_obj1 = NormSV(3, 100, 30)
        norm_obj2 = NormSV(4, 50, 10)
        norm_obj1.generate_norm_rasp()
        norm_obj2.generate_norm_rasp()
        hmm_obj = HMM(norm_obj1.rasp, norm_obj2.rasp)
        self.assertEqual(hmm_obj.X, norm_obj1.rasp)
        self.assertEqual(hmm_obj.Y, norm_obj2.rasp)

    def test_use_two_rasps_in_hmm(self):
        exp_obj1 = ExpSV(3, 0.1)
        norm_obj2 = NormSV(4, 50, 10)
        exp_obj1.generate_exp_rasp()
        norm_obj2.generate_norm_rasp()
        hmm_obj = HMM(exp_obj1.rasp, norm_obj2.rasp)
        self.assertEqual(hmm_obj.X, exp_obj1.rasp)
        self.assertEqual(hmm_obj.Y, norm_obj2.rasp)


if __name__ == '__main__':
    main()
