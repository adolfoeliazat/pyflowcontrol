#!/usr/bin/env python3

from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.son import SON
import json
import ast
# MongoDB definitions
MONGO_HOST = "roscoche.com"
MONGO_PORT = 27017
MONGO_USER = "roscoche"
MONGO_DB = "sentinela"
MONGO_PASS = input("Input MongoDB password for database {}: ".format(MONGO_DB))

connection = MongoClient(MONGO_HOST,MONGO_PORT)
db = connection[MONGO_DB]
db.authenticate(MONGO_USER, MONGO_PASS)
usuarios = db.usuarios.find()
#for person in usuarios:
#    print(person)
vetor2=[1,2,3]
codes=[]
codes.append("b'5935f9632a43f909db12ad05'")
pessoa = db.usuarios.find_one({'_id': ObjectId(str(codes[0])[2:-1])})
print(pessoa)
ExpectedCaracteristics = ast.literal_eval(pessoa['caracteristicas'])
print(ExpectedCaracteristics)
print(ExpectedCaracteristics['c1'])
#print(ExpectedCaracteristics['caracteristicas'])

#ExpectedCaracteristics = SON.to_dict(ExpectedCaracteristics['caracteristicas'])
#for x in ExpectedCaracteristics:
#    print(x)

carac1={}
if pessoa is not None:
    print(pessoa['nome'])
    carac1=ExpectedCaracteristics['caracteristicas']['c1']
    print(ExpectedCaracteristics['caracteristicas'])
    print('Caracteristicas: ' , carac1)
    print('LENGTH: ',str(len(ExpectedCaracteristics)))
    if 0 <= len(carac1) :
        stringcarac="{{'{}':{}}}".format('c1',str(vetor2))
        vetor=carac1
        #print("stringcarac " + stringcarac)
        #print("vetor: ",vetor)
        db.usuarios.update({'_id': ObjectId(str(codes[0])[2:-1])},{'$set': {'caracteristicas':stringcarac}})
else:
    print("Person not found on database")
