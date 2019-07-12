## Deploying models on IBM Cloud
Your IBM Account credentials will be needed.

## Training

```bash
# Spiral model
python parkinson_detector.py --modelname spiral

# Wave model
python parkinson_detector.py --modelname wave
```

## Deploy via Cloud Foundry

```bash
cd deployment/ibm-cloud

# login
cf login

# deploy
cf push
```

*WebApp should be lauches on IBM Cloud on a url with this pattern https://parkinson-xxxx-xxxx.mybluemix.net

#Python runtime in IBM-Cloud:
https://devcenter.heroku.com/articles/python-runtimes
