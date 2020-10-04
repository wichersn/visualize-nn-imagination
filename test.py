import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        metric_test_eval_datas = gen_data_batch(10000, 4)

        def test_metric(eval_datas, gen_boards, thresh, expected_min, expected_max):
            for i in range(2):
                print("eval_datas:")
                plt_boards(eval_datas[i])

                print("gen_boards:")
                plt_boards(gen_boards[i])

                print("=" * 40)

            metric_val = metric(eval_datas, gen_boards, thresh)

            print("metric_val", metric_val)

            assert (metric_val >= expected_min)
            assert (metric_val <= expected_max)

        print("Same data should be 1 ish, even when reordered")
        metric_test_gen_datas = np.stack(
            [metric_test_eval_datas[:, 0], metric_test_eval_datas[:, 2], metric_test_eval_datas[:, 3],
             metric_test_eval_datas[:, 1], metric_test_eval_datas[:, 4]], axis=1)
        test_metric(metric_test_eval_datas, metric_test_gen_datas, .99, 1, 1.1)

        print("Threshold works")
        assert (metric(metric_test_eval_datas, metric_test_gen_datas, .99) < metric(metric_test_eval_datas,
                                                                                    metric_test_gen_datas, .4))

        print("Only gets partial credit if repeating same state")
        metric_test_gen_datas = np.stack(
            [metric_test_eval_datas[:, 0], metric_test_eval_datas[:, 1], metric_test_eval_datas[:, 2],
             metric_test_eval_datas[:, 2], metric_test_eval_datas[:, 4]], axis=1)
        # It should get (1 + 1 + .5) / 3 = .83
        test_metric(metric_test_eval_datas, metric_test_gen_datas, .99, .83, .87)

        print("No credit for start or end state")
        metric_test_gen_datas = np.stack(
            [metric_test_eval_datas[:, 0], metric_test_eval_datas[:, 2], metric_test_eval_datas[:, 0],
             metric_test_eval_datas[:, 4], metric_test_eval_datas[:, 4]], axis=1)
        # It should get (1 + 0 + 0) / 3 = .33
        test_metric(metric_test_eval_datas, metric_test_gen_datas, .99, .33, .4)

        print("Combine metric works")
        combine_val = combine_metric(metric_test_eval_datas, metric_test_eval_datas, metric_test_gen_datas, .95)
        print(combine_val)
        assert (combine_val >= 1)
        assert (combine_val <= 1.1)


if __name__ == '__main__':
    unittest.main()
