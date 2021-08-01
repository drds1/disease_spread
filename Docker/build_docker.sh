docker build -f docker/Dockerfile -t test_container .
docker run --rm -it -v test_container bash