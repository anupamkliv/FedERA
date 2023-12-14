
from random import randint
from collections import OrderedDict
from concurrent import futures
import copy

##modify the verify function to consider the updated control variates also
def verify(clients, trained_model_state_dicts, save_dir_path, threshold = 0, updated_control_variates = None, server_model_state_dict = None):
    verification_dict = OrderedDict()
    config_dict = {"message": "verify"}

    ##if server_model_state_dict is not None then to each trained_model_state_dict, we need to add the server_model_state_dict
    if server_model_state_dict is not None:
        for i in range(len(trained_model_state_dicts)):
            for key in server_model_state_dict.keys():
                trained_model_state_dicts[i][key] += server_model_state_dict[key]
            
    for i, client in zip( range(len(clients)), clients):
        verification_dict[client.client_id] = {"client_wrapper_object": client, "model": trained_model_state_dicts[i], "control_variates": updated_control_variates[i]}
    client_ids = list(verification_dict.keys())
    client_ids_shuffled = random_derangement(client_ids)
    for i, client_id in zip( range(len(verification_dict)), verification_dict.keys() ):
        verification_dict[client_id]["assigned_client_id"] = client_ids_shuffled[i]

    with futures.ThreadPoolExecutor(max_workers = 20) as executor:
        result_futures = []
        for client_id, client_info in verification_dict.items():
            assigned_client_id = client_info["assigned_client_id"]
            assigned_client = verification_dict[assigned_client_id]["client_wrapper_object"]
            model_to_verify = client_info["model"]
            config_dict['client_id'] = client_id
            config_dict_s = copy.deepcopy(config_dict)
            result_futures.append(executor.submit(assigned_client.evaluate, model_to_verify, config_dict_s))


        verification_results = [result_future.result() for result_future in futures.as_completed(result_futures)]

        for index in range(len(verification_results)):
            verification_dict[verification_results[index]["client_id"]]["score"] = verification_results[index]["eval_accuracy"]


    selected_client_models, ignored_client_models, selected_control_variates = [], [], []
    for client_id, client_info in verification_dict.items():
        if client_info["score"] >= threshold:
            selected_client_models.append(client_info["model"])
            selected_control_variates.append(client_info["control_variates"])
            client_info["selected"] = True

        else:
            ignored_client_models.append(verification_dict[client_id]["model"])
            verification_dict[client_id]["selected"] = False

    #saves the client_id, its score, which client verified and fraction for the selected and ignored clients
    results_to_store = []
    for client_id, client_info in verification_dict.items():
        dict_to_store = {
            "client_id": client_id,
            "assigned_client_id": client_info["assigned_client_id"],
            "score": client_info["score"],
            "selected": client_info["selected"]
        }
        results_to_store.append(dict_to_store)

    selected_clients = [ client_dict for client_dict in results_to_store if client_dict["selected"] ]
    ignored_clients = [ client_dict for client_dict in results_to_store if not client_dict["selected"] ]
    num_of_selected_clients = len(selected_clients)
    num_of_ignored_clients = len(ignored_clients)
    num_of_total_clients = len(results_to_store)

    selected_info_dict = {
        "threshold": threshold,
        "selected": f"{num_of_selected_clients}/{num_of_total_clients}",
        "results": selected_clients
        }
    ignored_info_dict = {
        "threshold": threshold,
        "ignored": f"{num_of_ignored_clients}/{num_of_total_clients}",
        "results": ignored_clients
        }
    with open(f"{save_dir_path}/verification_selected_stats.txt", "a", encoding='UTF-8') as file:
        file.write( f"{selected_info_dict}\n" )
    with open(f"{save_dir_path}/verification_ignored_stats.txt", "a", encoding='UTF-8') as file:
        file.write( f"{ignored_info_dict}\n" )

    if server_model_state_dict is not None:
        for i in range(len(selected_client_models)):
            for key in server_model_state_dict.keys():
                selected_client_models[i][key] -= server_model_state_dict[key]

    return selected_client_models, selected_control_variates


def random_derangement(list_to_shuffle):
    for index1 in range(1, len(list_to_shuffle)):
        index2 = randint(0, index1 - 1) # nosec
        list_to_shuffle[index1], list_to_shuffle[index2] = list_to_shuffle[index2], list_to_shuffle[index1]
    return list_to_shuffle