#docker build -f Docker/Dockerfile -t test_container .

docker build -f Docker/Dockerfile --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa.pub)" -t test_container .


#docker run --rm -it -v test_container bash
docker run -it test_container bash
#docker run --rm -it test_container bash
#docker attach test_container

#delete all containers and images
#docker rm -vf $(docker ps -a -q)
#docker rmi -f $(docker images -a -q)
