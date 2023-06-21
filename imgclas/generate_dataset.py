import sqlite3
from enum import Enum
from datetime import datetime
import os
import hashlib
import multiprocessing
from multiprocessing import Pool
import pandas as pd
from fast_ml.model_development import train_valid_test_split
from skimage import io

etiqueta = Enum('etiqueta', ["location", "fruta", "variedad", "tamaño", "plato", "luz", "plano", "angulo"], start=0)

traduccion_etiqueta = { "Rio": "Rioja",
                        "Mad": "Madrid",

                        "M": "Manzana",
                        "P": "Pera",
                        "N": "Naranja",
                        "PL": "Platano",

                        "Fuji": "Fuji",
                        "Golden": "Golden",
                        "Granny Smith": "Granny Smith",

                        "Spb": ["Sin_Plato", "Blanco"],
                        "Spo": ["Sin_Plato", "Oscuro"],
                        "Spm": ["Sin_Plato", "Mantel"],
                        "Ppb": ["Postre", "NULL"],
                        "Pgb": ["Grande", "NULL"],

                        "int": "Interior",
                        "ext": "Exterior",

                        "al": "Alejado",
                        "me": "Medio",
                        "ce": "Cercano",

                        "sup": "Superior",
                        "cen": "Central",
                        "ver": "Vertical"}

directory = 'files'

class fruitVarietyError(Exception):
    "Raised when the fruit dont have that variety"
    pass

def db_connect():
    
    db = sqlite3.connect('/srv/imagesFruta/fruta.db')
    
    return db

def get_id_fruta(fruta):

    db = db_connect()
    cursor = db.cursor()
    #Obtengo las respuestas de cada intervalo de mas nuevas a mas antiguas
    cursor.execute('Select idfruta from frutas where fruta = "{}"'. format(fruta))
    
    return int(cursor.fetchall()[0][0])


def get_id_variedad(variedad):
    db = db_connect()
    cursor = db.cursor()
    #Obtengo las respuestas de cada intervalo de mas nuevas a mas antiguas
    cursor.execute('Select idvariedad from variedades where variedad = "{}"'. format(variedad))

    return int(cursor.fetchall()[0][0])

def make_insert_imagen(fecha, user, picture, hash, location, idfruta, idvariedad, tamaño, luz, plano, angulo, plato, superficie):
    try:
        db = db_connect()
        cursor = db.cursor()
        cursor.execute('INSERT INTO uploads_lab (date, user, picture, hash, location, idfruta, idvariedad, tamaño, luz, plano, angulo, plato, superficie) \
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?);', (fecha, user, picture, hash, location, idfruta, idvariedad, tamaño, luz, plano, angulo, plato, superficie))
        db.commit() #commit el insert
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)


def get_hash(path):
    md5_hash = hashlib.md5()
    with open(path,"rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            md5_hash.update(byte_block)
        #print(md5_hash.hexdigest())
        
    return md5_hash.hexdigest()

def check_variedad_fruta(idfruta, idvariedad):

    db = db_connect()
    cursor = db.cursor()
    #Obtengo las respuestas de cada intervalo de mas nuevas a mas antiguas
    cursor.execute('Select idfruta from variedades where idvariedad ="{}"'.format(idvariedad))
    if int(idfruta) != int(cursor.fetchall()[0][0]):
        raise fruitVarietyError

def validate_images(varietyDirectory):
    print("Hi from " + varietyDirectory)
    
    ERROR_IMAGES = list()
    for directory in os.listdir(varietyDirectory): # Para cada peso
        directory = varietyDirectory + directory + "/"
        #print(directory)
        for filename in os.listdir(directory): 
            #print(filename)
            f = os.path.join(directory, filename)
            splited = filename[0:-4].split('_') #Separamos por barraBaja, quitando .JPG
            #print(splited)
            #Recopilar informacion imagen
            try :
                fecha = datetime.now()
                user = "LAB" # "LAB" como usuario para las imagenes de laboratorio
                picture = filename # nombre de la imagen
                hash = get_hash(directory+"/"+filename) #Hash MD%
                location = traduccion_etiqueta[splited[etiqueta.location.value]]
                idfruta = get_id_fruta(traduccion_etiqueta[splited[etiqueta.fruta.value]])
                idvariedad = get_id_variedad(traduccion_etiqueta[splited[etiqueta.variedad.value]])
                tamaño = splited[etiqueta.tamaño.value]
                luz = traduccion_etiqueta[splited[etiqueta.luz.value]]
                plano = traduccion_etiqueta[splited[etiqueta.plano.value]]
                angulo = traduccion_etiqueta[splited[etiqueta.angulo.value]]
                plato = traduccion_etiqueta[splited[etiqueta.plato.value]][0]
                superficie = traduccion_etiqueta[splited[etiqueta.plato.value]][1]
                check_variedad_fruta(int(idfruta), int(idvariedad))
            except fruitVarietyError:
                print("Wrong variety: " + f)
                ERROR_IMAGES.append(f)
            except:
                #print(splited)
                ERROR_IMAGES.append(f)
            

    return ERROR_IMAGES

def validar_etiquetas(fruitDirectory):

    variety = list()
    for varietyDirectory in os.listdir(fruitDirectory):
        variety.append(fruitDirectory + varietyDirectory + "/")
    pool = multiprocessing.Pool()

    with Pool() as pool:
        with open('checkEtiquetas.log', 'w') as f: #Escribo en fichero etiquetas erroneas
            f.write("Etiquetas erroneas\n")
            for result in pool.map(validate_images, variety):
                string = ' '.join([str(image)+'\n' for image in result])
                f.write(string)
            f.close()
        #print(ERROR_IMAGES)
    pool.close()
    
def generate_dataset(fruitDirectory):

    # VALUES(date, user, picture, hash, location, idfruta, idvariedad, tamaño, luz, plano, angulo, plato, superficie);
    with open('../data/dataset_files/dataset.txt', 'w') as fw: #Escribo en fichero dataset
        for varietyDirectory in os.listdir(fruitDirectory): # Para cada variedad
            varietyDirectory = fruitDirectory + varietyDirectory + "/"
            print(varietyDirectory)
            for directory in os.listdir(varietyDirectory): # Para cada peso
                directory = varietyDirectory + directory + "/"
                #print(directory)
                for filename in os.listdir(directory): 
                    #print(filename)
                    f = os.path.join(directory, filename)
                    #checking if it is a file
                    if not os.path.isfile(f):
                        raise Exception("File Not found: " + str(f))
                    splited = filename[0:-4].split('_') #Separamos por barraBaja, quitando .JPG

                    try:
                        _ = io.imread(f)
                        #Recopilar informacion imagen
                        picture = filename # nombre de la imagen
                        tamaño = splited[etiqueta.tamaño.value]
                        
                        string = str(f) + '*' + str(tamaño.replace(",","."))+'\n'
                        fw.write(string)
                    except Exception as e:
                        print(f)

    fw.close()

    return 0

def dividir_dataset():
    
    df = pd.DataFrame(columns=['file', 'size'])
    f = open("../data/dataset_files/dataset.txt", "r")
    
    #Create pandas dataset
    for file in f.readlines():
        splited = file[0:-1].split('*')

        df = df.append(pd.DataFrame([splited], columns=['file', 'size']), ignore_index=True)

    
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target = 'size', 
                                                                            train_size=0.7, valid_size=0.1, test_size=0.2)
    
    with open('../data/dataset_files/train.txt', 'w') as fw: #Escribo en fichero Train
        for i in range(len(X_train)):
            string = X_train.iloc[i][0] + '*' + str(y_train.iloc[i]+ '\n')
            fw.write(string)
        fw.close()
        
    with open('../data/dataset_files/val.txt', 'w') as fw: #Escribo en fichero validation
        for i in range(len(X_valid)):
            string = X_valid.iloc[i][0] + '*' + str(y_valid.iloc[i]+ '\n')
            fw.write(string)
        fw.close()
        
    with open('../data/dataset_files/test.txt', 'w') as fw: #Escribo en fichero test
        for i in range(len(X_test)):
            string = X_test.iloc[i][0] + '*' + str(y_test.iloc[i]+ '\n')
            fw.write(string)
        fw.close()
    
generate_dataset("/srv/nextcloud/MANZANA/")
dividir_dataset()