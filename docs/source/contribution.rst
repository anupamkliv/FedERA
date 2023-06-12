.. _contribution:

**********************
Contribution to FedERA
**********************

Reporting bugs
--------------

To report bugs or request features, we utilize GitHub issues. If you come across a bug or have an idea for a feature, don't hesitate to open an issue.

If you encounter any problems while using this software package, please submit a ticket to the Bug Tracker. Additionally, you can post pull requests or feature requests.

Contributing to FedERA
----------------------

If you wish to contribute to the project by submitting code, you can do so by creating a Pull Request. By contributing code, you agree that your contributions will be licensed under `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`_.

We encourage you to contribute to the enhancement of **FedERA** or the implementation of existing FL methods within **FedERA**. The recommended method for contributing to **FedERA** is to fork the main repository on GitHub, clone it, and develop on a branch. Follow these steps:

1. Click on "Fork" to fork the project repository.

2. Clone your forked repository from your GitHub account to your local machine:
  
    .. code-block:: shell-session
        
        $ git clone https://github.com/anupamkliv/FedERA.git

    and then navigate to the FedLab directory using the command
    
    .. code-block:: shell-session
        
        $ cd FedERA

3. Create a new branch to save your changes using the command

    .. code-block:: shell-session
        
        $ git checkout -b my-feature
 
4. Develop the feature on your branch and use the command 

    .. code-block:: shell-session
        
        $ git add modified_files
   
    followed by 

    .. code-block:: shell-session
        
        $ git commit 

   to save your changes.

Pull Request Checklist
----------------------

- Please follow the file structure below for new features or create new file if there are something new.

    .. code-block:: shell-session
        FedERA
        ├── federa
        │   ├── client
        │   │   ├── src
        │   |   |   ├── client_lib
        │   |   |   ├── client
        │   |   |   ├── ClientConnection_pb2_grpc
        │   |   |   ├── ClientConnection_pb2
        │   |   |   ├── data_utils
        │   |   |   ├── distribution
        │   |   |   ├── get_data
        │   |   |   ├── net_lib
        │   |   |   ├── net
        │   │   └── start_client
        │   ├── server
        │   │   ├── src
        │   │   |   ├── algorithms
        │   │   |   ├── server_evaluate
        │   │   |   ├── client_connection_servicer
        │   │   |   ├── client_manager
        │   │   |   ├── client_wrapper
        │   │   |   ├── ClientConnection_pb2_grpc
        │   │   |   ├── ClientConnection_pb2
        │   │   |   ├── server_lib
        │   │   |   ├── server
        │   │   |   ├── verification
        │   │   └── start_server
        |   └── test
        |       ├── minitest
        |       └── misc
        │        
        └── test
            ├── misc
            ├── benchtest
            |   ├── test_results
            |   └── test_scalability
            └──unittest
                ├── test_algorithms
                ├── test_datasets
                ├── test_models
                └── test_modules
                