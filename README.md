## Objective : Build a model that will predict gender based on name, Containerize Machine Learning Application using Docker.

1. Build docker container 
### docker build -t gender-prediction -f Dockerfile .
2. Run docker container
### docker run --volume="$PWD:/usr/src/app" -p 5000:5000 -t gender-prediction
3. go to http://127.0.0.1:5000 to access the web-app


Alternatively:
1. you can train machine learning model from terminal using train_model.py
2. launch the web-app locally from the terminal using app.py 