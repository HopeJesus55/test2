from unittest import TestCase, main
from class_SV import SV, ExpSV, NormSV
from class_HMM import HMM


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
