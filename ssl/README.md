# SSL Configuration

## CFSSL Integration

This section provides instructions on using the CFSSL toolkit in conjunction with the files provided in this repository. To obtain the CFSSL toolkit, please visit the [CFSSL Website](https://cfssl.org/).

## File Customization

Please note that the files in this directory should be customized with your own details, particularly the `ca-config.json` and `ca-csr.json` files. While minimal modifications are sufficient for basic testing purposes, it is recommended to update these files to align with your specific requirements.

## Certificate Authority Generation

To generate the Certificate Authority (CA) files, execute the following command:

```sh
 cfssl gencert -initca ca-csr.json | cfssljson -bare ca
```

This command will generate the `ca.pem` and `ca-key.pem` files. These files are utilized for generating client and server certificates. The `ca.pem` file is used for mutual verification between clients and servers.

## Client Certificate Generation

To generate a client certificate, use the following command:

```sh
cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json client-csr.json | cfssljson -bare client
```

This command will generate the `client.pem` and `client-key.pem` files.

**_Note:_** A warning message may appear during the execution of this command, indicating the lack of a "hosts" field in the certificate. However, for client certificates, the absence of this field is acceptable as they are not intended for use as servers.

## Server Certificate Generation

To generate a server certificate, execute the following command:

```sh
cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -hostname=<your server hostname> server-csr.json | cfssljson -bare server
```

This command will generate the `server.pem` and `server-key.pem` files.

## Acknowledgements

The code and information in this directory were developed with the help of the repository [joekottke/python-grpc-ssl](https://github.com/joekottke/python-grpc-ssl), which provided valuable guidance in implementing the encryption functionality.
