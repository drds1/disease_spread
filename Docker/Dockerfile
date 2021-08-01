# syntax=docker/dockerfile:1.0.0-experimental
FROM amancevice/pandas:0.23.4-python3-slim

WORKDIR /Main

#COPY requirements.txt ./
#COPY main.py ./
#
#RUN mkdir disease_spread
#COPY disease_spread/ /Main/disease_spread
#
#RUN mkdir results
#COPY results/ /Main/results/
#
#RUN pip install --no-cache-dir -r requirements.txt

# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# add credentials on build
ARG SSH_PRIVATE_KEY
ENV SSH_PRIVATE_KEY=$SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "testing ssh key"
RUN echo $SSH_PRIVATE_KEY
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN echo "$(cat /root/.ssh/id_rsa)"


# make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
RUN ssh-keyscan github.org >> /root/.ssh/known_hosts
RUN ssh-keyscan gitlab.com >> /root/.ssh/known_hosts

# clone the repo
#RUN git clone git@github.com:drds1/disease_spread.git


#cd to repo directory
#RUN cd /Main/disease_spread
#RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python /main/main.py"]
