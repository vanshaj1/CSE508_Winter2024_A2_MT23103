# Citations:- https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=The%20TF%2DIDF%20of%20a,multiplying%20TF%20and%20IDF%20scores.&text=Translated%20into%20plain%20English%2C%20importance,between%20documents%20measured%20by%20IDF. , https://www.kaggle.com/code/uthamkanth/beginner-tf-idf-and-cosine-similarity-from-scratch , https://www.analyticsvidhya.com/blog/2023/02/lets-start-with-image-preprocessing-using-skimage/#Image_Preprocessing_-_Image_Flipping, https://www.hackersrealm.net/post/extract-features-from-image-python, https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/
 
import os
import csv
import cv2
import nltk
import string
import skimage
import requests
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform
from skimage import io
from keras.models import Model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from skimage.exposure import adjust_gamma
from keras.applications.vgg16 import VGG16
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from io import BytesIO
from skimage.transform import rotate
import pickle
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.preprocessing import normalize
#****************** Data Structure *********************
# Array of Dictionaries

   #[ Dictionary 
      # Product Id
      # Image URL
      # Image Features ]

# 2d matrix of terms vs documents 


#*************** Array of Dictionaries **************

# Products
products = []

# images
images = []


#****************** Image processing ******************

#************************ Image pre processing *****************
def image_pre_processing(url):
    # loading image
    data = requests.get(url).content 
    image_name = url.split("/")[-1:]
    f = open('images/'+ str(image_name[0]),'wb') 
    f.write(data) 
    f.close() 

    image_Matrix = []
    try:
        image_Matrix = io.imread('images/'+ str(image_name[0]))
    except:
        return None


    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image_Matrix, cv2.COLOR_BGR2RGB)

    # Image rotation parameter
    center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
    angle = 30
    scale = 1
    
    # getRotationMatrix2D creates a matrix needed for transformation.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # We want matrix for rotation w.r.t center to 30 degree without scaling.
    image = cv2.warpAffine(image_rgb, rotation_matrix, (image_Matrix.shape[1], image_Matrix.shape[0]))

    # brighten the image
    image = adjust_gamma(image,gamma=0.5,gain=1)

    # Resize image
    image = cv2.resize(image, (224, 224) )
    image = image.reshape(1,224,224,3)

    return image


#************************* Image features Extraction ******************************
def get_image_features(preprocessed_img):
    #***************** Feature extraction *******************
    # load the model
    feature_extraction_model = VGG16()

    # restructure the model
    model = Model(inputs=feature_extraction_model.inputs, outputs=feature_extraction_model.layers[-2].output)

    # extract features
    features = model.predict(preprocessed_img, verbose=0)
    features = normalize(features)

    return features



#***************** Text Processing *******************

#************************* Text Pre-Processing *********************
def text_pre_processing(review):
    #******* Lowercase conversion *******
    lowercase_review = review.lower()

    #******* Tokenization ***************
    tokenize_review = word_tokenize(lowercase_review)

    #******* Punctuation Removal ********
    punctuation_free_review = []
    for token in tokenize_review:
        token = token.strip()
        translator = str.maketrans('', '', string.punctuation)
        newToken = token.translate(translator)
        punctuation_free_review.append(newToken)

    #****** Stopwords Removal ***********
    stopwords_free_review = []
    stopwordsList = stopwords.words("english")
    for word in punctuation_free_review:
        word = word.strip()
        if not word in stopwordsList:
            stopwords_free_review.append(word)

    #****** Stemming ********************
    porterStemmer = PorterStemmer()
    stemmed_review = []
    for word in stopwords_free_review:
        word = porterStemmer.stem(word)
        stemmed_review.append(word)

    #****** Lemmetization ***************
    wordnetLemmatizer = WordNetLemmatizer()
    Lemmtized_review = []
    for word in stemmed_review:
        word = wordnetLemmatizer.lemmatize(word)
        Lemmtized_review.append(word)

    #****** Whitespace removal **********
    whitespace_removed_review = []
    for word in Lemmtized_review:
        newToken = word.strip()
        if(newToken != ''):
            whitespace_removed_review.append(newToken)

    return whitespace_removed_review





#**************************** TF-TDF Calcuation *************************

def unique_terms(documents):
    uni_terms = []
    for document in documents:
        for term in document:
            if term not in uni_terms:
                uni_terms.append(term)
    return uni_terms


def Term_Frequency(documents,terms_list):
    noOfDocuments = len(documents)
    noOfUniqueTerms = len(terms_list)

    term_doc_matrix = pd.DataFrame(np.zeros((noOfDocuments, noOfUniqueTerms)), columns=terms_list)

    for i in range(noOfDocuments):
        for term in documents[i]:
            term_doc_matrix[term][i] = term_doc_matrix[term][i] + 1
    
    return term_doc_matrix


def Inverse_Document_Frequency(documents,terms_list):
    idf = {}

    NoOfDocuments = len(documents)

    for term in terms_list:
        appearedInNoOfDoc = 0

        for document in documents:
            if term in document:
                appearedInNoOfDoc += 1
        
        # idf[term] =  np.log10(NoOfDocuments / appearedInNoOfDoc)

        if appearedInNoOfDoc > 0:
            idf[term] = np.log10(NoOfDocuments / appearedInNoOfDoc)
        else :
            idf[term] = 0

    return idf



    

def TF_IDF(term_doc_matrix,IDF,noOfDocuments, terms_list):
    tf_idf = term_doc_matrix.copy()

    for i in range(noOfDocuments):
        for term in terms_list:
            tf_idf[term][i] = tf_idf[term][i] * IDF[term]
    
    return tf_idf



#*************************** Cosine similarity ***************************

#**************** Text cosine similarity *******************
def unique_terms_Query(document):
    uni_terms = []
    for term in document:
        if term not in uni_terms:
            uni_terms.append(term)
    return uni_terms

def term_frequency_Query(query,term_list):
    tf = {}
    for term in term_list:
        for q_term in query:
            if q_term == term:
                if q_term in tf:
                    tf[q_term] = tf[q_term] + 1
                else :
                    tf[q_term] = 1
        if term not in tf:
            tf[term] = 0
    return tf

def Inverse_Document_Frequency_Query(documents,query_term_list,terms_list):
    idf = {}

    NoOfDocuments = len(documents)

    for term in terms_list:
        appearedInNoOfDoc = 0
        for query_term in query_term_list:
            if query_term == term:
                for document in documents:
                    if term in document:
                        appearedInNoOfDoc += 1
                
                # idf[term] =  np.log10(NoOfDocuments / appearedInNoOfDoc)

                if appearedInNoOfDoc > 0:
                    idf[term] = np.log10(NoOfDocuments / appearedInNoOfDoc)
                else :
                    idf[term] = 0
            else: 
                idf[term] = 0

    return idf


def TF_IDF_QUERY(tf,IDF, terms_list):
    tf_idf = []
    # print(tf)
    # print(IDF)
    for term in terms_list:
        # print(term)
        tf_idf.append(tf[term] * IDF[term])
    return tf_idf


def compute_cosine_similarity(Query,Doc):
    score = cosine_similarity(Query, Doc)
    return score[0][0]

def ranking_basis_cosine_similarity_text(tf_idf_d,tf_idf_q):
    cosine_similarity_dict = {}
    ranked_result = []

    for i in range(len(tf_idf_d)):
        X = np.array(tf_idf_d.iloc[i]).reshape(1,-1)
        # print(X)
        Y = [tf_idf_q]
        score = compute_cosine_similarity(X,Y)
        cosine_similarity_dict[products[i]["id"]] = score
        products[i]["text_score"] = score

    cosine_similarity_dict_d  = OrderedDict(sorted(cosine_similarity_dict.items(), key=lambda item: item[1], reverse=True))

    # print(cosine_similarity_dict_d)
    i = 0
    for key in cosine_similarity_dict_d:
        if( i == 3):
            break
        ranked_result.append(key)
        i += 1
    
    return ranked_result



#*************** Image Cosine similairity **********************
def ranking_basis_cosine_similarity_image(query_features):
    ranked_result = []

    for image in images:
        features = image["image_features"]
        score = compute_cosine_similarity(features,query_features)
        image["image_score"] = score

    
    images_d  = sorted(images, key=lambda d: d['image_score'],reverse=True)

    for image in images_d:
        product_temp = {}
        for product in products:
            if product["id"] == image["id"]:
                product_temp = product
                break
        product_temp["images_score"].append(image["image_score"])


    i = 0
    for image in images_d:
        if i == 3:
            break
        if image["id"] not in ranked_result:
            ranked_result.append(image["id"])
            i += 1
    
    return ranked_result



#**************************** Input Output ******************************
def reading_CSV():
    with open('A2_Data.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        i = 0
        for line in csvFile:
            if(i == 0):
                i += 1
                continue

            product = {}
            product["id"] = line[0]
            product["image_url"] = line[1].strip('][').split(', ')
            product["image_review"] = line[2]
            product["images_score"] = []
            product["text_score"] = 0
            product["composite_score"] = 0
            products.append(product)
            # print(product)

def fill_images_feature_data():
    for product in products:
        print(product)
        for url in product["image_url"]:
            image = {}
            image["id"] = product["id"]
            image["image_url"] = product["image_url"]
            print(url)
            url = url[1:len(url)-1]
            pre_processed_img = image_pre_processing(url)
            if(pre_processed_img is None):
                continue
            feature = get_image_features(pre_processed_img)
            image["image_features"] = feature
            images.append(image)


def ranking_based_composite_score(ranked_ids_image,ranked_ids_text):
    composite_score_ids = {}
    ranked_results = []

    for id in ranked_ids_image:
        for product in products:
            if product["id"] == id:
                composite_score = product["images_score"][0] + product["text_score"]
                composite_score_ids[id] = composite_score
                product["composite_score"] = composite_score
    
    for id in ranked_ids_text:
        for product in products:
            if product["id"] == id:
                composite_score = product["images_score"][0] + product["text_score"]
                composite_score_ids[id] = composite_score
                product["composite_score"] = composite_score

    composite_score_ids_d  = OrderedDict(sorted(composite_score_ids.items(), key=lambda item: item[1], reverse=True))

    for key in composite_score_ids_d:
        ranked_results.append(key)
    
    return ranked_results


def load_data_from_file(filename):
    file_data = open(filename, 'rb')
    data = pickle.load(file_data)
    file_data.close()
    return data

def save_data_to_file(filename, data):
    data_file= open(filename, 'wb')
    pickle.dump(data, data_file)  
    data_file.close()



def mainProcess():
    reading_CSV()   
    # print(products)
    # fill_images_feature_data()

    #*************************************** TESTING ***********************************
    # pre_processed_img = image_pre_processing('https://images-na.ssl-images-amazon.com/images/I/71J74QaTycL._SY88.jpg')
    # if(pre_processed_img is None):
    #     print("None")
    #     return
    # feature = get_image_features(pre_processed_img)
    # print(feature)
    #************************************************************************************

    # print(images)
    # image_feature_data_file = open('image_features', 'wb')
    # pickle.dump(images, image_feature_data_file)  
    # image_feature_data_file.close()

    documents = []
    
    for product in products:
        document = text_pre_processing(product["image_review"])
        documents.append(document)
    
    terms_list = unique_terms(documents)
    # term_doc_matrix = Term_Frequency(documents,terms_list)
    idf = Inverse_Document_Frequency(documents,terms_list)
    # tf_idf = TF_IDF(term_doc_matrix,idf,len(documents),terms_list)

    #**************************** Testing ******************************
    # tf_idf = [{"vanshaj":"sharma"}]
    # print(tf_idf)
    #*******************************************************************

    # tf_idf_data_file = open('IF_IDF', 'wb')
    # pickle.dump(tf_idf, tf_idf_data_file)  
    # tf_idf_data_file.close()

    tf_idf = load_data_from_file('IF_IDF')

    global images
    images_features_data = open('image_features', 'rb')
    images = pickle.load(images_features_data)
    images_features_data.close()

    print("Image and Text Query Input: ")
    query_url = input("Image: ") 
    query_review = input("Review: ")

    query_pre_processed_img = image_pre_processing(query_url)
    query_img_feature = get_image_features(query_pre_processed_img)

    query_review_doc = text_pre_processing(query_review)
    # print("1.",query_review_doc)
    # query_term_list = unique_terms_Query(query_review_doc)
    query_tf = term_frequency_Query(query_review_doc,terms_list)
    # query_idf = Inverse_Document_Frequency_Query(documents,query_term_list,terms_list)
    query_tf_idf = TF_IDF_QUERY(query_tf,idf,terms_list)
    # print(query_tf_idf)

    ranked_ids_image = ranking_basis_cosine_similarity_image(query_img_feature)
    ranked_ids_text = ranking_basis_cosine_similarity_text(tf_idf,query_tf_idf)
    ranked_ids_composite = ranking_based_composite_score(ranked_ids_image,ranked_ids_text)

    
    save_data_to_file("ranked_id_images",ranked_ids_image)
    save_data_to_file("ranked_ids_text",ranked_ids_text)
    save_data_to_file("ranked_ids_composite",ranked_ids_composite)

    ranked_ids_image = load_data_from_file("ranked_id_images")
    ranked_ids_text = load_data_from_file("ranked_ids_text")
    ranked_ids_composite = load_data_from_file("ranked_ids_composite")
    


    print("USING IMAGE RETRIEVAL:- ")
    for id in ranked_ids_image:
        for product in products:
            if id == product["id"]:
                print("Image URL: " + str(product["image_url"]))
                print("Review: "+ str(product["image_review"]))
                print("Cosine similarity of images: "+ str(product["images_score"]))
                print("Cosine similarity of text: "+ str(product["text_score"]))
                print("\n")

    print("USING TEXT RETRIEVAL:- ")
    for id in ranked_ids_text:
        for product in products:
            if id == product["id"]:
                print("Image URL: " + str(product["image_url"]))
                print("Review: "+ str(product["image_review"]))
                print("Cosine similarity of images: "+ str(product["images_score"]))
                print("Cosine similarity of text: "+ str(product["text_score"]))
                print("\n")

    print("USING COMPOSITE RANK RETRIEVAL:- ")
    for id in ranked_ids_composite:
        for product in products:
            if id == product["id"]:
                print("Image URL: " + str(product["image_url"]))
                print("Review: "+ str(product["image_review"]))
                print("Cosine similarity of images: "+ str(product["images_score"]))
                print("Cosine similarity of text: "+ str(product["text_score"]))
                print("Composite score: "+ str(product["composite_score"]))
                print("\n")

mainProcess()


































