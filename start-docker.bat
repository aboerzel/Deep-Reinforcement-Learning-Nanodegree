d:
cd d:\development\drlnd
docker run --rm -it -p 8888:8888 --name drlnd -v %cd%/deep-reinforcement-learning:/workspace -u root drlnd
