.. _installation:

Installation 
============

Follow this procedure to prepare the environment and install **FedERA**:

1. Install a Python 3.8 (>=3.6, <=3.9) virtual environment using venv.
   
 See the `Venv installation guide <https://docs.python.org/3/library/venv.html>`_ for details.

2. Create a new Virtualenv environment for the project.

   .. code-block:: console

      python3 -m venv venv

3. Activate the virtual environment.

   .. code-block:: console

      source venv/bin/activate

4. Install the **FedERA** package.

    A. Install the **stable version** with pip:

        .. code-block:: console

            $ pip install feder==$version$
   
    B. Install the **latest version** from GitHub:

        1. Clone the **FedERA** repository:
        
            .. code-block:: console
            
                $ git clone https://github.com/anupamkliv/FedERA.git
                $ cd FedERA

        2. Install dependencies:
        
            .. code-block:: console
            
                $ pip install -r requirements.txt

FedERA with Docker
==================

Follow this procedure to build a Docker image of **FedERA**:

.. note::

   The purpose of the Docker edition of **FedERA** is to provide an isolated environment complete with the prerequisites to run. Once the execution is finished, the container can be eliminated, and the computation results will be accessible in a directory on the local host.

1. Install Docker on all nodes in the federation.

 See the `Docker installation guide <https://docs.docker.com/engine/install/>`_ for details. 

2. Check that Docker is running properly with the *Hello World* command:

    .. code-block:: console

      $ docker run hello-world
      Hello from Docker!
      This message shows that your installation appears to be working correctly.
      ...
      ...
      ...

3. Build the Docker image of **FedERA**:

      .. code-block:: console
   
         $ docker build -t federa .

4. Run the Docker image of **FedERA**:

      .. code-block:: console
   
         $ docker run federa
