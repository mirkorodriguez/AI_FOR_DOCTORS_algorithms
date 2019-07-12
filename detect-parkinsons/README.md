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

*WebApp should be launched on IBM Cloud. The url will have this pattern: https://parkinson-xxxx-xxxx.mybluemix.net

#Python runtime in IBM-Cloud:
https://devcenter.heroku.com/articles/python-runtimes

*Code for parkinson_detector.py was modified from the original one you can find here https://www.pyimagesearch.com/2019/04/29/detecting-parkinsons-disease-with-opencv-computer-vision-and-the-spiral-wave-test/
