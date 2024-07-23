# Collaborative Filtering Integration to the YaCy Decentralized Search Engine
## General information:
- In order to access the servers from a outside network, used ports (see ``` docker-compose.yml ```) needs to be forwarded.
- Since the socket is running on a separate port, the peers that will collaborate should be added to the whitelist.
- For this project, the socket port should be YaCy port + 100.
- There is no check for packet loss or loss of data during the model transmission. This is a future work.
- Due to docker implementation, config support is currently changed with env variables. Update config module so that if
there are no env variables use the config file instead.

## Requirements

- **Python Version**: 3.12.x
- Docker

## Setup Instructions For Using The System

1. Download the ``` docker-compose.yml ``` file or clone the repository.
2. Change your working directory to the folder where the ``` docker-compose ``` file is.
3. In your router settings, go to port forwarding and forward the ports that your YaCy peer is running in and the socket port (YaCy port + 100)
4. Run ``` docker-compose up ```
5. In your browser go to ``` localhost:4000 ```

## Setup Instructions For Testing The Backend Only

### 1. Clone the Repository

First, clone the repository to your local machine using `git`:

```sh
git clone https://github.com/damla-jpg/yacy_cf_backend
```

### 2. Navigate to the Project Directory

Change your working directory to the project's folder:

```sh
cd yacy_cf_backend
```

### 3. Set Up Virtual Environment

Create a virtual environment to manage dependencies:

```sh
python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

### 4. Install Dependencies

Install the required Python packages:

```sh
pip install -r requirements.txt
```

### 5. Configure Docker

Ensure Docker is installed and running on your system. Build and run the Docker container for YaCy:

```sh
docker build -t yacy .
docker run -d -p 8090:8090 yacy
```

### 6. Configure Ports and Network

Forward the necessary ports to allow external network access. Ensure that the socket port of each peer is known to other peers.

Ports to be forwarded:
- Yacy port
- Socket Port

### 7. Run the Application

Execute the main application script:

```sh
python main.py
```

### 8. Accessing the Results

Once the setup is complete and the application is running, you can access the predictions by making a GET request to the following URL:

```
http://localhost:[FLASK_PORT]/api/fetch_predictions
```

Replace `[FLASK_PORT]` with the actual port number specified in your configuration.

### 9. Viewing the Algorithm in Action (if you don't want to use Docker)

To see the collaborative filtering algorithm in action, follow these steps:

1. **Install and Run the Front-End Interface**:
   - Clone the front-end repository from GitHub:

     ```sh
     git clone https://github.com/damla-jpg/yacy_cf_integration
     ```

   - Navigate to the front-end directory:

     ```sh
     cd yacy_cf_integration
     ```

   - Install the necessary dependencies:

     ```sh
     npm install
     ```

   - Start the front-end application:

     ```sh
     npm start
     ```

2. **Access the Front-End Interface**:
   - Open your web browser and navigate to the address provided by the front-end application (usually `http://localhost:3000`).

By following these steps, you'll be able to interact with the front-end interface and observe the collaborative filtering algorithm in real-time.

## Usage

### Configuring Peers

To configure peers for collaborative filtering, add them to your whitelist.

### Monitoring and Logging

Logs are generated to help monitor the application's performance and to troubleshoot any issues. Logs can be found in the `logs` directory.

## Future Work

- **Socket Integration**: Integrate socket port configuration directly into YaCy for seamless peer communication.
- **Packet Loss Detection**: Implement mechanisms to detect and handle packet loss during model transmission.
- **Whitelist Configuration**: Once integrated into YaCy, the need for a whitelist will be eliminated.

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Contact

For any questions or support, please open an issue on the GitHub repository or contact the project maintainers directly at damlaural@gmail.com .

---

Thank you for using and contributing to the Collaborative Filtering Integration to the YaCy Decentralized Search Engine.

## Acknowledgements
- Collabrative Filtering Algorithm that has been used and altered in this project is from Krishna Shukla and can be found from

```sh
https://gitlab.com/ucbooks/dht/-/tree/main?ref_type=heads
```
