from unittest import TestCase, main
from class_SV import SV, ExpSV, NormSV


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



if __name__ == '__main__':
    main()