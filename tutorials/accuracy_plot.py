import os
import matplotlib.pyplot as plt


def read_values(txt_path):
    """
    Reads accuracy and round values from a text file.

    Args:
        txt_path (str): Path to the text file.

    Returns:
        tuple: Accuracy and round values as lists.
    """
    accuracy = []
    rounds = []
    with open(txt_path, encoding='UTF-8') as f:
        for line in f.readlines():
            line_data = line.split(',')
            acc = float(line_data[1].split(':')[1])
            r = int(line_data[2].split(':')[1].split('}')[0])
            accuracy.append(acc)
            rounds.append(r)
    return accuracy, rounds


def plot_round_vs_accuracy_1(algorithm_values, niids):
    """
    Plots round vs accuracy graphs for different NIID datasets and algorithms.

    Args:
        algorithm_values (list): List of tuples containing algorithm name and results for different NIIDs.
        niids (list): List of number of clients for which FL was performed.
    """

    for niid in range(len(niids)):
        plt.figure()
        plt.title(f'NIID Dataset {niids[niid]} vs Accuracy')  # Varying title based on the NIID dataset
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')

        for algorithm in range(len(algorithm_values)):
            accuracy, rounds = algorithm_values[algorithm][niid][1][0], algorithm_values[algorithm][niid][1][1]
            algorithm_name = algorithm_values[algorithm][0][0]
            plt.plot(rounds, accuracy, markersize=5, label=algorithm_name)

        plt.legend()
        plt.savefig(f'./media/NIID-{str(niids[niid])}_niid_vs_accuracy.png')
        plt.clf()

        del accuracy
        del rounds


def plot_round_vs_accuracy_2(algorithm_values, niids):
    """
    Plots round vs accuracy graphs for different algorithms and a single NIID dataset.

    Args:
        algorithm_values (list): List of tuples containing algorithm name and results for different NIIDs.
        niids (list): List of number of clients for which FL was performed.
    """

    for i, algorithm_value in enumerate(algorithm_values):
        plt.figure()

        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        for niid in range(len(niids)):
            plt.title(f'Round vs Accuracy for {str(algorithm_value[niid][0])}')
            #print(algorithm_value[niid][0])
            accuracy, rounds = algorithm_value[niid][1][0], algorithm_value[niid][1][1]
            plt.plot(rounds, accuracy,  markersize=5, label=f'NIID = {niid+1}')
        plt.legend()
        plt.savefig(f'./media/round_vs_accuracy_{str(algorithm_value[niid][0])}.png')


def plot_niid_vs_accuracy(algorithm_values, niids):
    """
    Plots NIID vs accuracy graph for different algorithms.

    Args:
        algorithm_values (list): List of tuples containing algorithm name and results for different NIIDs.
        niids (list): List of number of clients for which FL was performed.
    """
    plt.figure()
    plt.title('NIID vs Accuracy')
    plt.xlabel('NIID')
    plt.ylabel('Accuracy')

    for algorithm_data in algorithm_values:
        algorithm_name = algorithm_data[0][0]
        accuracy = []

        for niid in range(len(niids)):
            accuracy.append(algorithm_data[niid][1][0][-1])

        plt.plot(niids, accuracy, 'o--', markersize=5, label=algorithm_name)

    plt.legend()
    plt.savefig('./media/Niid_vs_Accuracy.png')
    plt.clf()


if __name__ == '__main__':

    # Check if the result directory exists
    results_path = '../server_results'
    if not os.path.exists(results_path):
        raise Exception("The result directory is not found")
    else:

        dataset_name = 'FashionMNIST'
        algorithm_names = ['fedavg', 'fedadam', 'fedavgm', 'fedadagrad', 'fedyogi']

        # Get the list of number of clients
        niids = sorted([int(i) for i in os.listdir(os.path.join(results_path, dataset_name, algorithm_names[0]))])

        # Get the FL results for each algorithm and each number of clients
        algorithm_values = []
        for algorithm_name in algorithm_names:
            algorithm_path = os.path.join(results_path, dataset_name, algorithm_name)
            algorithm_niids = os.listdir(algorithm_path)
            algorithm_niids = sorted([int(i) for i in algorithm_niids])
            algorithm_values.append([])
            for niid in algorithm_niids:
                niid_path = os.path.join(algorithm_path, str(niid))
                niid_runs = os.listdir(niid_path)
                latest_run = sorted(niid_runs, reverse=True)[0]
                results_file_path = os.path.join(niid_path, latest_run, 'FL_results.txt')
                values = read_values(results_file_path)   # Assumes read_values function is defined elsewhere
                algorithm_values[-1].append((algorithm_name, values))

        # Plot the results
        # Assumes plot_round_vs_accuracy_1 function is defined elsewhere
        plot_round_vs_accuracy_1(algorithm_values, niids)
        # Assumes plot_round_vs_accuracy_2 function is defined elsewhere
        plot_round_vs_accuracy_2(algorithm_values, niids)
        plot_niid_vs_accuracy(algorithm_values, niids) # Assumes plot_round_vs_accuracy_2 function is defined elsewhere
