.. _encryption:

**********
Encryption
**********

In the FedERA framework, encryption plays a crucial role in ensuring secure communication between the client and server during the Federated Learning process. This section provides guidance on generating and configuring the necessary certificates for TLS/SSL encryption.

TLS Basics
==========

To understand the encryption process, it's essential to grasp the fundamentals of TLS/SSL and chains of trust. TLS/SSL operates based on a transitive trust model, where trust in a certificate authority (CA) extends to the certificates it generates. Web browsers and operating systems have a "Trusted Roots" certificate store, automatically trusting certificates from public certificate authorities such as Let's Encrypt or GoDaddy.

In the case of FedERA, we establish our own CA and need to inform the client about the CA certificate for trust verification. Additionally, the server certificate must contain the exact server name the client connects to for validation.

Generate Certificates
=====================

For the purpose of this example, we will set up a basic PKI Infrastructure using CloudFlare's CFSSL toolset, specifically the `cfssl` and `cfssljson` tools. You can download these tools from `here <https://pkg.cfssl.org>`_ .

The `ssl` directory contains configuration files that can be modified, but for demonstration purposes, they can also be used as-is.

Generate CA Certificate and Config
----------------------------------

To generate the CA certificate and configuration, navigate to the `ssl` directory and run the following command:

.. code-block:: shell-session

    $ cd FedERA/ssl
    $ cfssl gencert -initca ca-csr.json | cfssljson -bare ca


This command generates the `ca.pem` and `ca-key.pem` files. The `ca.pem` file is used by both the client and server for mutual verification.

Generate Server and Client Certificates
---------------------------------------

Server Certificate
~~~~~~~~~~~~~~~~~~

To generate the server certificate and key pair, run the following command in the `ssl` directory:

.. code-block:: shell-session

    $ cd FedERA/ssl
    $ cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -hostname='127.0.0.1,localhost' server-csr.json | cfssljson -bare server


This command creates the server certificate and key pair to be used by the server during TLS/SSL encryption. Note that you can modify the `hostname` parameter to match the name or IP address of the server on your network.

Client Certificate
~~~~~~~~~~~~~~~~~~

To generate the client certificate and key pair, use the following command in the `ssl` directory:

.. code-block:: shell-session

    $ cd FedERA/ssl
    $ cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json client-csr.json | cfssljson -bare client


When generating the client certificate and key pair, a warning message may appear regarding the absence of a "hosts" field. This warning is expected and acceptable since the client certificate is only used for client identification, not server identification.

TLS Server Identification and Encryption
========================================

In FedERA, the client trusts the certificate authority certificate, which subsequently enables trust in the server certificate. This is similar to how web browsers handle certificates, where pre-installed public certificate authority certificates establish trust.

For one-way trust verification (client verifies server identity but not vice versa), the server does not necessarily need to present the CA certificate as part of its certificate chain. The server only needs to present enough of the certificate chain for the client to trace it back to a trusted CA certificate.

In the FedERA framework, the gRPC server can be configured for SSL using the following code snippet:
----------------------------------------------------------------------------------------------------

On server side
~~~~~~~~~~~~~~

.. code-block:: python

    if configurations['encryption']==1:
            # Load the server's private key and certificate
            keyfile = configurations['server_key']
            certfile = configurations['server_cert']
            private_key = bytes(open(keyfile).read(), 'utf-8')
            certificate_chain = bytes(open(certfile).read(), 'utf-8')
            # Create SSL/TLS credentials object
            server_credentials = ssl_server_credentials([(private_key, certificate_chain)])
            server.add_secure_port('localhost:8214', server_credentials)

On client side
~~~~~~~~~~~~~~

.. code-block:: python

    if config["encryption"] == 1:
                ca_cert = 'ca.pem'
                root_certs = bytes(open(ca_cert).read(), 'utf-8')
                credentials = grpc.ssl_channel_credentials(root_certs)
                #create new gRPC channel to the server
                channel = grpc.secure_channel(ip_address, options=[
                    ('grpc.max_send_message_length', -1),
                    ('grpc.max_receive_message_length', -1)
                    ], credentials=credentials)

Acknowledgments
===============
This code and information were developed with the help of the repository `jottoekke/python-grpc-ssl <https://github.com/joekottke/python-grpc-ssl>`_, which provided valuable guidance in implementing the encryption functionality.
