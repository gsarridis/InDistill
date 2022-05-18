import pickle

def print_results(results_path, topx=100):

    with open(results_path + '_retrieval.pickle', 'rb') as f:
        results = pickle.load(f)

    # print("Retrieval top-1",  100 * results['cifar10']['raw_precision'][0])
    map_c = 100 * results['cifar10']['map']
    top50_c = 100 * results['cifar10']['raw_precision'][topx-1]

    with open(results_path + '_retrieval_e.pickle', 'rb') as f:
        results = pickle.load(f)
    map_e = 100 * results['cifar10']['map']
    top50_e = 100 * results['cifar10']['raw_precision'][topx-1]

    line = ' mAP (e) = %3.2f, mAP (c) = %3.2f, top-100 pr. (e) = %3.2f, top-100 pr. (c) = %3.2f' % (map_e, map_c, top50_e, top50_c)
    print(line)


if __name__ == '__main__':
    
    print("Teacher CIFAR10:")
    print_results(results_path='./results/scores/teacher_baseline_cifar10')

    print("Auxiliary CIFAR10:")
    print_results(results_path='./results/scores/auxiliary_cifar10')


    print("Student CIFAR10:")
    print_results(results_path='./results/scores/student_baseline_cifar10')

    print("InDistill-CLS CIFAR10:")
    print_results(results_path='./results/scores/student_cifar10_indistill_pkt_cls')
