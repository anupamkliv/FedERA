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


.. list-table:: Server Configuration Options
   :widths: 25 45 20
   :header-rows: 1
   
   * - Argument
     - Description
     - Default
   * - ``algorithm``
     - specifies the aggregation algorithm
     - ``fedavg``
   * - ``clients``
     - specifies number of clients selected per round
     - ``1``
   * - ``fraction``
     - specifies fraction of clients selected
     - ``1``
   * - ``rounds``
     - specifies total number of rounds
     - ``1``
   * - ``model_path``
     - specifies initial server model path
     - ``initial_model.pt``
   * - ``epochs``
     - specifies client epochs per round
     - ``1``
   * - ``accept_conn``
     - determines if connections accepted after FL begins
     - ``1``
   * - ``verify``
     - specifies if verification module runs before rounds
     - ``0``
   * - ``threshold``
     - specifies minimum verification score
     - ``0``
   * - ``timeout``
     - specifies client training time limit per round
     - ``None``
   * - ``resize_size``
     - specifies dataset resize dimension
     - ``32``
   * - ``batch_size``
     - specifies dataset batch size
     - ``32``
   * - ``net``
     - specifies network architecture
     - ``LeNet``
   * - ``dataset``
     - specifies dataset name
     - ``MNIST``
   * - ``niid``
     - specifies data distribution among clients
     - ``1``
   * - ``carbon``
     - specifies if carbon emissions tracked at client side
     - ``0``
   * - ``encryption``
     - specifies whether to use SSL encryption or not
     - ``0``
   * - ``server_key``
     - specifies path to server key certificate
     - ``server-key.pem``
   * - ``server_cert``
     - specifies path to server certificate
     - ``server.pem``


Starting the Clients
--------------------

The clients are started by running the following command in the root directory of the framework:

.. code-block:: console

    python federa.client.start_client

Arguments that can be passed to the clients are:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Client Configuration Options
   :widths: 25 45 20
   :header-rows: 1
   
   * - Argument
     - Description
     - Default
   * - ``server_ip``
     - specifies server IP address
     - ``localhost:8214``
   * - ``device``
     - specifies device
     - ``cpu``
   * - ``encryption``
     - specifies whether to use SSL encryption or not
     - ``0``
   * - ``ca``
     - specifies path to CA certificate
     - ``ca.pem``
   * - ``wait_time``
     - specifies time to wait before reconnecting to the server
     - ``30``
    