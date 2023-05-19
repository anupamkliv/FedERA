.. _running:

*******************************
Running the Server and Clients
*******************************

Starting the Server
-------------------

The server is started by running the following command in the root directory of the framework:

.. code-block:: console

    python -m federa.server.start_server

Arguments that can be passed to the server are:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* --algorithm: This argument specifies the aggregation algorithm to use. The type of this argument is string, and the default value is "fedavg".
* --clients: This argument specifies the number of clients to select for each communication round. The type of this argument is integer, and the default value is 2.
* --fraction: This argument specifies the fraction of clients to select from the available pool of clients. The type of this argument is float, and the default value is 1. This argument accepts values between 0 and 1 inclusive.
* --rounds: This argument specifies the total number of communication rounds to perform. The type of this argument is integer, and the default value is 2.
* --model_path: This argument specifies the path of the initial server model's state dictionary. The default value is "initial_model.pt".
* --epochs: This argument specifies the number of epochs each client should perform in each communication round. The type of this argument is integer, and the default value is 1.
* --accept_conn: This argument determines whether connections will be accepted even after FL has begun. The type of this argument is integer, and the default value is 1 (meaning that connections will be accepted).
* --verify: This argument specifies whether the verification module should be run before each communication round. The type of this argument is integer, and the default value is 0 (meaning that verification is disabled).
* --threshold: This argument specifies the minimum score that clients must have in a verification round (if verification is enabled). The type of this argument is float, and the default value is 0. This argument accepts values between 0 and 1 inclusive.
* --timeout: This argument specifies the time limit that each client has when training during each communication round. The type of this argument is integer, and the default value is None (meaning that there is no time limit).
* --resize_size: This argument specifies the resize dimension for the dataset. The type of this argument is integer, and the default value is 32.
* --batch_size: This argument specifies the batch size for the dataset. The type of this argument is integer, and the default value is 32.
* --net: This argument specifies the network architecture to use. The type of this argument is string, and the default value is "LeNet".
* --dataset: This argument specifies the name of the dataset to use. The type of this argument is string, and the default value is "FashionMNIST". If the value of this argument is "CUSTOM", the algorithm will use a local dataset.
* --niid: This argument specifies the type of data distribution among clients. The type of this argument is integer, and the default value is 1. The value of this argument should be either 1 or 5.
* --carbon: This argument specifies whether carbon emissions need to be tracked at the client side. The type of this argument is integer, and the default value is 0 (meaning that carbon emissions will not be tracked).


Starting the Clients
--------------------

The clients are started by running the following command in the root directory of the framework:

.. code-block:: console

    python federa.client.start_client

Arguments that can be passed to the clients are:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    --device cuda
    --server_ip localhost:8214
    