import json
import requests
import datetime
import os

from requests.exceptions import HTTPError
from google.oauth2.credentials import Credentials
from google.cloud.firestore import Client

# daten von smartcoaster
nfcUid = "0582861b27d100"
drink = "water"
weight = 200
date = datetime.datetime.now()
isNewDrink = False
spectroData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
container = "mug"

# Use the google verify password REST api to authenticate and generate user tokens
def sign_in_with_email_and_password(api_key, email, password):
    FIREBASE_REST_API = "https://identitytoolkit.googleapis.com/v1/accounts"
    request_url = "%s:signInWithPassword?key=%s" % (FIREBASE_REST_API, api_key)
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"email": email, "password": password, "returnSecureToken": True})
    
    resp = requests.post(request_url, headers=headers, data=data)
    # Check for errors
    try:
        resp.raise_for_status()
    except HTTPError as e:
        raise HTTPError(e, resp.text)
        
    return resp.json()

def handleDbCommuication(nfcUid, drink, weight, date, isNewDrink, spectroData, container):
    # Credentials are loaded from environment for publication safety. Set these before running.
    apiKey = os.environ.get("SMARTCOASTER_FIREBASE_API_KEY", "<set-api-key>")
    email = os.environ.get("SMARTCOASTER_FIREBASE_EMAIL", "<set-email>")
    password = os.environ.get("SMARTCOASTER_FIREBASE_PASSWORD", "<set-password>")
    projectId = os.environ.get("SMARTCOASTER_FIREBASE_PROJECT", "<set-project>")
    
    # We use the sign_in_with_email_and_password function from https://gist.github.com/Bob-Thomas/49fcd13bbd890ba9031cc76be46ce446
    response = sign_in_with_email_and_password(apiKey, email, password)

    # Use google.oauth2.credentials and the response object to create the correct user credentials
    creds = Credentials(response['idToken'], response['refreshToken'])

    # Use the raw firestore grpc client instead of building one through firebase_admin
    db = Client(projectId, creds)

    try:
        # get userUid 
        nfcDoc = db.collection('nfcCards').document(nfcUid).get()
        userUid = nfcDoc.get('userUid')
        
        # add drink 
        drinkObject = {
            "userUid": userUid,
            "drink": drink,
            "weight": weight,
            "date": date,
            "newDrink": isNewDrink
        }
        response = db.collection("drinks").add(drinkObject)
        drinkDocId = response[1].id
        
        # add model data
        modelData = {
            "drink": drink,
            "weight": weight,
            "container": container,
            "drinkDocId": drinkDocId,
            "spectroData": spectroData
        }
        response = db.collection("modelData").add(modelData)

    except: 
        print("error: something went wrong!")


handleDbCommuication(nfcUid, drink, weight, date, isNewDrink, spectroData, container)