"""
This file containes helper functions that are used to build the app
"""
import re
import cv2
import numpy as np
import config as config
import os
from collections import defaultdict
import re
import pdb
import pandas as pd
import spacy
import string
import nltk
import pytesseract
from pdf2image import convert_from_path
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from operator import itemgetter
from collections import Counter
import joblib

nlp = spacy.load('fr_core_news_md')

class Infer():
    def __init__(self, path, filetype = "pdf"):
        self.path = path
        vect = joblib.load(os.path.join(config.MODEL_PATH, "vectorizer.pk"))
        self.mlb = joblib.load(os.path.join(config.MODEL_PATH, "binarizer.pk"))
        ovr_rf_model = joblib.load(os.path.join(config.MODEL_PATH, "OVR_RF_model.sav"))
        self.filetype = filetype
        self.pipeline = Pipeline([('tfidf',vect),
                 ('clf',ovr_rf_model),])

    def infer_attributes(self):

        if self.filetype == "pdf":
            pages = convert_from_path(self.path, dpi=200)
            pdf2im = ConvertPDF2Images(self.path)

            for page in pages:
                pdf2im.save(page)

            images = pdf2im.read()
        else:
            images = cv2.imread(self.path)

        info = defaultdict(list)
        meth = 0
        score = 0
        # 3 because we have 3 methods of thresholding
        while score<4 and meth<3:
            text = ""

            for image in images:
                ip = ImageProcessing(image)
                proc_image, index = ip.process(meth)
                text += str(((pytesseract.image_to_string(
                    proc_image, lang='fra'))))

            text_xn = text.replace('-\n', '')
            text_xn = text_xn.replace('\n', ' ')
            #print(text)
            lookup = LookUp(text)
            info['text'].append(text)
            info['first'].append(lookup.getfirst())
            info['last'].append(lookup.getlast())

            medecin, most_common_medecin = lookup.keyword_positive_lookbehind([
                "Dr ", "Docteur ", "correspondants ", "copie ", "D\. ",
                "médecin traitant ", "destinataires \: ", "copie à \: "
            ],
                                                         name=True)
            info["medecin"].append(most_common_medecin)
            personne, most_common_personne = lookup.keyword_positive_lookbehind([
                "M\. ", "Mme\. ", "Mr ", "Mme ", "monsieur ", "madame ",
                "Madame ", "Monsieur ", "Mll ", "mademoiselle ", "l'enfant",
                "patient ",
            ],
                                                          match_till="né",
                                                          name=True)
            surname, most_common_surname = lookup.keyword_positive_lookbehind(["Nom \: "])
            forename, most_common_forename = lookup.keyword_positive_lookbehind(["Prénom \: "])

            if forename and surname:
                name = " ".join([most_common_forename[0][1], most_common_surname[0][1]])
                tuple_n = (forename[0][0], name)
                most_common_personne.append(tuple_n)

            elif surname:
                most_common_personne = surname

            info["personne"].append(most_common_personne)

            patient, most_common_patient = lookup.keyword_positive_lookbehind([
                "copie à \: ", "destinataires \: ", "concernant \: ",
                "double à \: "
            ])
            info["patient"].append(most_common_patient)
            objet, most_common_objet = lookup.keyword_positive_lookbehind(
                ["objet", "objet\:", "objet\: "])
            info["objet"].append(most_common_objet)

            digit = lookup.digit_info_lookup(text_xn)[0]
            info["dates"].append(digit["dates"])
            info["birthdates"].append(digit["birthdates"])
            info["age"].append(digit["age"])

            score = self.scoring(most_common_medecin)
            score = self.scoring(most_common_personne)
            score = self.scoring(digit["dates"])
            score = self.scoring(digit["birthdates"])
            scores.append(score)

            if not score==4:
                # reinitialisation
                score =0

            meth += 1

        df_info = pd.DataFrame(info)
        best_score_idx = np.argmax(scores)

        return df_info.loc[best_score_idx]

    def infer_topic(self, X ):
        X_trans = self.pipeline.named_steps["tfidf"].transform(X)
        y_pred_tfidf = self.pipeline.predict(X)
        y_pred_proba_tfidf = self.pipeline.predict_proba(X)
        y_pred_new_tfidf = get_best_tags(y_pred_tfidf, y_pred_proba_tfidf)
        y_tags = self.mlb.inverse_transform(y_pred_new_tfidf)

        return y_tags

    def run(self):
        X = self.infer_attributes()
        y = self.infer_topic(X['text_n'])
        X["topic_n"] = ",".join(y)

        return X

    def scoring(self, x, score):
        if x:
            score+=1
        return score

    def get_string(self, x):

        if len(x)>0:
            return x[0][1]
        else:
            return ""

def get_best_tags(y_pred, y_pred_proba, n_tags=2):
    """
    assign at least one tag to y_pred that only have 0

    Parameters:
    -----------
    y_pred: np array
        multilabel predicted y values

    y_pred_proba: np array
        multilabel predicted proba y values

    n_tags: int
        number of non-zero tags

    Returns:
    --------
    y_pred: np array
        new y_pred for evaluation purpose
    """
    y_pred_copy = y_pred.copy()
    idx_y_pred_zeros  = np.where(y_pred_copy.sum(axis=1)==0)[0]
    #idx_y_pred_zeros  = np.where(y_pred_copy.sum(axis=1)<n_tags)[0]
    best_tags = np.argsort(
        y_pred_proba[idx_y_pred_zeros])[:, :-(n_tags + 1):-1]

    for i in range(len(idx_y_pred_zeros)):
        y_pred_copy[idx_y_pred_zeros[i], best_tags[i]] = 1

    return y_pred_copy

def levenshteinDistance(s1, s2):
    """
    DP implementation of Levenshtein Distance
    """

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1],
                                distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]

def match_checking(truth, pred, e_distance=2):
    """
    if truth and pred are withing a certain number of distance given by
    e_distance, considered matched
    """

    dist = levenshteinDistance(truth, pred)

    if dist<=e_distance:
        return True
    else:
        return False

def process_date(x):
    """
    convert dates to date time type
    """

    if not (pd.isnull(x) or x==0):
        if isinstance(x, str):
            rep_x = x.replace(".","/")
            rep_x = rep_x.replace(" ","/")
            new_str_date = []

            for idx, d in enumerate(rep_x.split("/")):
                if idx==0:
                    if len(d)==1:
                        new_str_date.append("0"+d)
                    else:
                        new_str_date.append(d)
                elif idx==1:
                    if len(d)==1:
                        new_str_date.append("0"+d)
                    else:
                        new_str_date.append(d)
                else:
                    new_str_date.append(d)

            str_date = "".join(new_str_date)

        else:
            str_date = str(int(x))

        if len(str_date)==4:

            if int(str_date[2:])<config.REPL_YEAR:
                prefix_year = "20"
            else:
                prefix_year = "19"
            str_date = "0" + str_date[:1] +"0"+ str_date[1:2] + prefix_year+ str_date[2:]
            resp = "/".join([str_date[:2], str_date[2:4], str_date[4:]])

        elif len(str_date)==5 or len(str_date)==7:

            if int(str_date[3:])<config.REPL_YEAR:
                prefix_year = "20"
            else:
                prefix_year = "19"

            if len(str_date)==5 and int(str_date[0])>3:
                str_date = "0" + str_date[:1] + str_date[1:3] + prefix_year+ str_date[3:]

            elif len(str_date)==5 and int(str_date[0])<=3:
                str_date = str_date[:2] + "0" + str_date[2:3] + prefix_year+ str_date[3:]

            elif len(str_date)==7 and int(str_date[0])>3:
                str_date = "0" + str_date[1:]

            elif len(str_date)==7 and int(str_date[0])<=3:
                str_date = str_date[:1] + "0" + str_date[1:]
           #resp = "/".join([str_date[:1], str_date[1:3], str_date[3:]])
            resp = "/".join([str_date[:2], str_date[2:4], str_date[4:]])

        elif len(str_date)==6 or len(str_date)==8:

            if int(str_date[4:])<config.REPL_YEAR:
                prefix_year = "20"
            else:
                prefix_year = "19"

            if len(str_date)==6 and not (str_date[-4:-2]=="20" or str_date[-4:-2]=="19"):
                str_date = str_date[:2] + str_date[2:4] + prefix_year + str_date[4:]

            elif len(str_date)==6:
                str_date = "0" +str_date[:1] +"0"+ str_date[1:]

            resp = "/".join([str_date[:2], str_date[2:4], str_date[4:]])

        else:
            resp = None
    else:
        resp = None

    try:
        df_resp = pd.to_datetime(resp , format="%d/%m/%Y")

    except ValueError:
        df_resp = None

    return df_resp


def initialize_dict(path_info, element_array):

    for el in element_array:
        if el in ["type", "specialty"]:
            path_info[el] = []
        else:
            path_info[el] = None


def separate_path(path, delimiter="/"):
    temp = re.sub(config.DIR_PATH + "", "", path)

    path_array = path.split(delimiter)
    path_info = defaultdict(list)
    path_xext = path_array[-1].split(".")[0]
    path_elements = path_xext.split("_")

    initialize_dict(path_info, config.ELEMENT_ARRAY)

    for i, element in enumerate(path_elements):

        if element:
            if i == 0:
                path_info["doctor"] = element

            elif i == 1:
                path_info["patient"] = element

            elif element[0] == "S":
                path_info["specialty"].append(element[1:])

            elif element[0] == "E":
                path_info["emit"] = element[1:]

            elif element[0] == "T":
                path_info["type"].append(element[1:])

            elif element[0] == "L":
                path_info["part"].append(element[1:])

            elif element[0] == "A" and element[1:].isdigit():
                path_info["age"] = element[1:]

            elif element.isdigit():
                if int(element[-4:]) < 2008:
                    path_info["ddn"] = element

                else:
                    path_info["dde"] = element

            elif i == len(path_elements) - 1:
                if not path_info["type"]:
                    path_info["type"].append(element)


    dirpath = "/".join(path_array[:-1])
    type_folder = re.sub("\d ", "", path_array[-2]).lower()

    filename = path_array[-1]
    #file_name_norm = file_name.lower().replace(type_folder, "")
    subdirpath = "".join(dirpath.split(config.DIR_PATH + "/")[1::])
    path_info["subdirpath"] = subdirpath
    path_info["filename"] = filename
    path_info["abs_path"] = os.path.join(dirpath, filename)

    path_info["filename_n"] = filename.lower()
    path_info["dirpath"] = dirpath
    #file_name_norm = re.sub(type_folder, "", file_name)
    return path_info

def DFS(df, root):
    """
    Depth first search of the file system to generate the path to each file.
    """

    visited = set()
    stack = []
    stack.extend([root])

    while stack:
        root = stack.pop()

        if root not in visited:
            visited.add(root)

            if os.path.isfile(root):
                path_info = separate_path(root)
                df = df.append(path_info, ignore_index=True)

            else:
                subfolder_list = [
                    x for x in os.listdir(root)
                    if x not in [".DS_Store", ".gitkeep"]
                ]
                folder_list = [root + "/" + x for x in subfolder_list]
                stack.extend(folder_list)

    return df

class ConvertPDF2Images():

    def __init__(self, path):
        self.info = path
        self.image_counter = 1

    def save(self, page):
        path_info = separate_path(self.path, delimiter="/")
        sub_path_array = path_info["sub_dirpath"].split("/")
        counter = 0
        temp_folder = config.JPEG_DIR_PATH +"/"

        while counter<len(sub_path_array):
            temp_folder +=  sub_path_array[counter]+"/"

            if not os.path.exists(temp_folder):
                os.mkdir(temp_folder)

            counter +=1

        filename =  config.JPEG_DIR_PATH +"/"+ path_info["sub_dirpath"] + "/"+ path_info["filename"] + "_page_"+ str(self.image_counter)+".jpg"

        # Save the image of the page in system

        page.save(filename, 'JPEG')
        self.image_counter += 1

    def read(self):
        """
        return images for each file
        """
        filelimit = self.image_counter-1
        text = ""
        images = []
        # Iterate from 1 to total number of pages
        for i in range(0, filelimit):

            filename = config.JPEG_DIR_PATH +"/"+ self.info["subdirpath"] + "/"+ self.info["filename"] + "_page_"+str(i+1)+".jpg"

            image = cv2.imread(filename)

            images.append(image)

        return images

def getMostCommon(phrases):
    """
    get most common term in phrases
    """
    if phrases:
        idx, names = list(zip(*phrases))

        return [(0,Counter(names).most_common(1)[0][0])]

    return []

class LookUp():

    def __init__(self, text):
        self.text = text
        self.pt = ProcessText()

    def deiterator(self,infodict, digit=False):
        phrases = []

        for match in infodict:
            phrase = match.group(0)

            if not digit:
                bool_filter = filtering(phrase, config.FILTER_KEYWORDS)

                if bool_filter:
                    p_phrase = " ".join(self.pt.tokenize_stopword_removal(phrase, text=False))
                    tuple_phrase= (match.start(), p_phrase.strip())
                    phrases.append(tuple_phrase)
            else:
                tuple_phrase= (match.start(), phrase.strip())
                phrases.append(tuple_phrase)

        return phrases


    def getfirst(self):
        """
        get first word and its position
        """

        pattern = r"^\s*([^\s]+)"
        search = re.search(pattern , self.text )

        if search is not None:
            search_list = [(search.start(), search.group(0))]
        else:
            search_list = [(None, None)]

        return search_list

    def getlast(self):
        """
        get last word and its position
        """

        #pattern = r"^\s*([^\s]+)|\b([^\s]+)(?=\S$)"
        pattern = r"\b([^\s]+)(?=\S$)"
        search = re.search(pattern , self.text )

        if search is not None:
            search_list = [(search.start(), search.group(0))]
        else:
            search_list = [(None, None)]

        #search_list = self.deiterator(search)

        return search_list

    def keyword_lookup(self, keywords):
        """
        find matches with keywords using regex and return words that
        follow those keywords

        Parameters:
        -----------
        text : string
            text body

        keywords : numpy array
            list of keywords
        """

        pattern = r"(?:" +"|".join(keywords) + "(\w+)"
        search = re.finditer(pattern , self.text, flags=re.IGNORECASE )
        search_list = self.deiterator(search)
        most_common_term = getMostCommon(search_list)

        return search_list, most_common_term

    def keyword_positive_lookbehind(self, keywords, match_till = None, name = False):
        """
        find matches with keywords using regex and return words that
        follow those keywords

        Parameters:
        -----------
        text : string
            text body

        keywords : numpy array
            list of keywords
        """
        search_list_match_till = []
        search_list = []

        if match_till is not None:
            match_pattern = ".+?(?=" + match_till + ")"
            pattern =""

            for kw in keywords[:-1]:
                pattern += r"(?:(?<=" + kw + "))("+ match_pattern +")|"

            pattern +=r"(?:(?<=" + keywords[-1] + "))("+ match_pattern +")"

            search = re.finditer(pattern , self.text, flags=re.IGNORECASE )
            search_list_match_till = self.deiterator(search)

        if name:
            # match the next two words
            #match_pattern = r"\b[\w]+\b.*?\b[\w]+\b"
            match_pattern = r".+\b"

        else:
            # this match the word that is immediately after
            match_pattern = r"[^\s]+"

        pattern =""

        for kw in keywords[:-1]:
            pattern += r"(?:(?<=" + kw + "))("+ match_pattern +")|"

        pattern +=r"(?:(?<=" + keywords[-1] + "))("+ match_pattern +")"

        search = re.finditer(pattern , self.text, flags=re.IGNORECASE )
        search_list_temp = self.deiterator(search)

        if search_list_match_till:
            for i, ielement in search_list_match_till:
                for j, jelement in search_list_temp:
                    # relaxation
                    if j-i<=3:
                        search_list.append((i,ielement))
                    else:
                        search_list.append((j,jelement))
        else:
            search_list = search_list_temp

        most_common_term = getMostCommon(search_list)

        return search_list, most_common_term

    def digit_info_lookup(self, text):
        """
        lookup meaningful digits in the document : dates, age, birthdates,
        return a dictionary of matches
        """
        resp ={}
        search_list = {}
        most_common = {}

        #resp["dates"] = re.finditer(r"(?<=le )\d+\/\d+\/\d{2,4}", self.text)
        resp["dates"] = re.finditer(r"\d+[\/\.]\d+[\/\.]\d{2,4}", text)

        repl_month = {"janvier":"01","février":"02", "mars":"03","avril":"04","mai":"05",
                      "juin":"06", "juillet":"07", "août":"08","septembre":"09",
                      "octobre":"10", "novembre":"11","décembre":"12"}
        rep = dict((re.escape(k), str(v)) for k, v in repl_month.items())
        #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pat_date = re.compile("|".join(rep.keys()))

        pattern = r"\d+\s(?:"

        for idx, month in repl_month.items():
            if int(month)<12:
                pattern += str(idx) +"|"

        pattern += str(list(repl_month.keys())[-1]) + r")\s\d{2,4}"

        resp["fulldates"] = re.finditer(pattern, text, flags=re.IGNORECASE )
        #resp["birthdates"] = re.finditer(r"(?:(?<=née\sle\s)(\d+\/\d+\/\d{2,4}))|(?:(?<=né\sle\s)(\d+\/\d+\/\d{2,4}))", self.text)

        resp["age"] = re.finditer(r"(?:(?<=âgée\sde\s)(\d+)\sans)|(?:(?<=âgé\sde\s)(\d+)\sans)", text)
        dates = []
        birthdates = []

        for idx, date in self.deiterator(resp["dates"], digit=True):

            n_date = process_date(date)

            if n_date is not None:
                if n_date>config.CUR_YEAR:
                    dates.append((idx, n_date))

                elif n_date<=config.CUR_YEAR and n_date>config.BEGIN_YEAR:
                    birthdates.append((idx, n_date))

        for idx, date in self.deiterator(resp["fulldates"], digit=True):
                        # use these three lines to do the replacement
            r_date = pat_date.sub(lambda m: rep[re.escape(m.group(0))], date.lower())
            n_date = process_date(r_date)

            if n_date is not None:
                if n_date>config.CUR_YEAR:
                    dates.append((idx, n_date))

                elif n_date<=config.CUR_YEAR and n_date>config.BEGIN_YEAR:
                    birthdates.append((idx, n_date))

        search_list["dates"] = dates
        search_list["birthdates"] = birthdates
        search_list["age"] = self.deiterator(resp["age"], digit=True)

        most_common["dates"] = getMostCommon(dates)
        most_common["birthdates"] = getMostCommon(birthdates)
        most_common["age"] = getMostCommon(self.deiterator(resp["age"], digit=True))

        return search_list, most_common

def filtering(text, setKeywords, n_word_lim = 4):
    """
    return True if no unnecessary words foung in the text
    or when it's less than 5 words
    """
    words = [w for w in text.split()]

    if len(words)>n_word_lim:
        return False

    for word in words:

        bool_res = np.sum(any(i in word for i in setKeywords))

        if bool_res:
            return False

    return True


class ImageProcessing():

    def __init__(self, image):

        self.image = image
        self.proc_image = self.preprocess()

    def preprocess(self):
        # do this only once
        proc_image = self.gray_scaled(self.image)
        proc_image = self.contrast_building(proc_image)
        #proc_image = self.deskew(proc_image)

        return proc_image

    def process(self, index):
        """

        """

        proc_image, index = self.thresholding(self.proc_image, i=index)

        return proc_image, index

    @staticmethod
    def gray_scaled(image):
        """
        convert image to gray-scaled image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray

    @staticmethod
    def contrast_building(gray, params = None):
        """
        take in grayed imaged and apply adaptive histogram equalizer to build
        an image with higher contrast
        """

        if params is None:
            params = {"clipLimit":2.0, "tileGridSize":(8,8)}

        clahe = cv2.createCLAHE(**params)#clipLimit=2.0, tileGridSize=(8,8))

        contrast = clahe.apply(gray)

        return contrast

    @staticmethod
    def deskew(image):
        """
        take in gray image and deskew image so that the document
        will be upright.
        """
        image_not = cv2.bitwise_not(image)

        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        thresh = cv2.threshold(image_not, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)

        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        return rotated

    @staticmethod
    def thresholding(image, i = 0):
        """
        a successive of threholding is applied if the info is not found, to be
        used in a loop
        """

        if i == 0:
            thresh = cv2.threshold(image, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        elif i == 1:
            thresh = cv2.adaptiveThreshold(image, 255,
                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 12)

        elif i == 2:
            thresh = cv2.adaptiveThreshold(image, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 91, 12)

        return thresh, i

def check_similar_tags(a, b, G_tags):
    """
    check if tag a belongs to the same group as tag b

    Parameters:
    -----------
    a: string
        tag a

    b: string
        tag b

    sim_tags: networkx graph
        graph of tags

    Returns:
    --------
    boolean
        true if tag a and tag b belong to the same group
    """
    # throw errors in cases where element is not in the graph
    try:
        a_edge_list = list(nx.descendants(G_tags, a))
    except NetworkXException:
        a_edge_list = []

    try:
        b_edge_list = list(nx.descendants(G_tags, b))
    except NetworkXException:
        b_edge_list = []

    if (b in a_edge_list) and (a in b_edge_list):
        return True
    else:
        return False


def model_evaluate(y_true,
             y_pred,
             binarizer=None,
             G_tags=None,
             score_threshold=0.5,
             l_print_errors=False,
             l_deduplication=False):
    """
    returns accuracy score. In the case of crossval, returns errors in cross
    evaluation

    Parameters:
    -----------
    X: numpy array, or scipy sparse array
        entry features

    y: numpy array,
        multilabel y true labels

    classifier: scikit learn model
        trained model

    binarizer: multi label binarizer object
        to inverse binarized object to feature names

    G_tags: networkx graph
        used in tags regrouping

    score_threshold: float
        score threshold to decide if to consider as an error

    l_print_errors: boolean
        True if print errors

    l_deduplication: boolean
        remove duplicated tags in true and pred labels

    Returns:
    --------
    accuracy_score: float

    errors: array
        errors
    """

    f1 = 0.
    errors = []

    if binarizer is None:
        raise "Binarizer is None"

    if False:
        if G_tags is None:
            raise "G_tags is None"

    for index in range(len(y_true)):

        y_pred_temp = y_pred[index].reshape(1, -1)
        y_true_temp = y_true[index].reshape(1, -1)

        y_true_tag = binarizer.inverse_transform(y_true_temp)
        y_pred_tag = binarizer.inverse_transform(y_pred_temp)

        index_true = np.where(y_true_temp.flatten() == 1)[0]
        index_pred = np.where(y_pred_temp.flatten() == 1)[0]
        union_index = list(set(index_true).union(set(index_pred)))

        if False:
            if l_deduplication:
                # check for duplicated true values
                for p in range(len(y_true_tag[0])):
                    for q in range(p + 1, len(y_true_tag[0])):
                        if check_similar_tags(y_true_tag[0][p], y_true_tag[0][q],
                                              G_tags):
                            y_true_temp[0, index_true[q]] = 0

                # check for duplicated predictions
                for p in range(len(y_pred_tag[0])):
                    for q in range(p + 1, len(y_pred_tag[0])):
                        if check_similar_tags(y_pred_tag[0][p], y_pred_tag[0][q],
                                              G_tags):
                            y_pred_temp[0, index_pred[q]] = 0

            y_true_tag_dedup = binarizer.inverse_transform(y_true_temp)
            y_pred_tag_dedup = binarizer.inverse_transform(y_pred_temp)

            for i in range(len(y_true_tag_dedup)):
                for j in range(len(y_pred_tag_dedup)):
                    if check_similar_tags(y_true_tag_dedup[0][i], y_pred_tag_dedup[0][j],
                                          G_tags):
                        y_true_temp[0, union_index[i]] = 1
                        y_pred_temp[0, union_index[j]] = 1

        #y_true_eval = y_true_temp[0, union_index]
        #y_pred_eval = y_pred_temp[0, union_index]

        score_f1 = f1_score(y_true_temp.flatten(), y_pred_temp.flatten())
        f1 += score_f1
        #jac_sim = jaccard_similarity_score(
        #    y_true_eval, y_pred_eval, normalize=False)
        # if only half of the total are mislabeled,
        # then we consider that it's correct
        if l_print_errors:
            if score_f1 < score_threshold:
                errors.append((binarizer.inverse_transform(y_true_temp),
                               binarizer.inverse_transform(y_pred_temp)))

        #if jac_sim >= n_correct:
        #    accuracy += 1.

    if l_print_errors:
        return f1 / y_true.shape[0], errors
    else:
        # normalize accuracy to the number of samples
        return f1 / y_true.shape[0]

class Evaluation():
    def __init__(self, medecin, personne, dates, birthdates, verbose=False):
        self.medecin = medecin
        self.personne = personne
        self.dates = dates
        self.birthdates = birthdates
        # insert 0 as default value
        self.score_values = defaultdict(int)#lambda:0)
        self.verbose = verbose
        # start with a different score if validation value is absent
        if medecin is None:
            self.score_values["medecin"] = 1

        elif personne is None:
            self.score_values["patient"] = 1

        elif pd.isnull(dates):
            self.score_values["dde"] = 1

        elif pd.isnull(birthdates):
            self.score_values["ddn"] = 1

        self.score_medecin = False
        self.score_personne = False
        self.score_dates = False
        self.score_birthdates = False

    def score(self, p_medecin, p_personne, p_dates, p_birthdates):

        if p_medecin:
            for idx, med in p_medecin:
                if not self.score_medecin and self.forename_name_invert(
                        self.medecin, med, doctor=True):
                    if self.verbose:
                        print("Medecin OK")
                    self.score_medecin = True
                    self.score_values["medecin"] = 1

        if p_personne:
            for idx, personne in p_personne:
                if not self.score_personne and self.forename_name_invert(
                        self.personne, personne):
                    if self.verbose:
                        print("Personne OK")
                    self.score_personne = True
                    self.score_values["patient"] = 1

        if p_dates:
            for idx, date in p_dates:
                if not self.score_dates and self.dates == date:
                    if self.verbose:
                        print("Date OK")
                    self.score_dates = True
                    self.score_values["dde"] = 1

        if p_birthdates:
            for idx, birthdate in p_birthdates:
                if not self.score_birthdates and self.birthdates == birthdate:
                    if self.verbose:
                        print("Birthdates OK")
                    self.score_birthdates = True
                    self.score_values["ddn"] = 1

    @staticmethod
    def forename_name_invert(t_name, p_name, doctor=False):

        sep_name = p_name.lower().split(' ')

        a_name = " ".join(sep_name)
        b_name = " ".join(sep_name[::-1])

        if doctor:
            # if doctor's family name is present, return true
            t_name_f = t_name.split(' ')[-1]
            l_check = False

            for s_name in sep_name:
                l_check = match_checking(t_name_f, s_name)

                if l_check:
                    return True

            return l_check

        else:
            l_check = []
            t_name_f = t_name.split(' ')

            for s_name in sep_name:
                for t_n in t_name_f:
                    l_check.append(match_checking(t_n, s_name))

            if np.sum(l_check) == len(sep_name):
                return True
            else:
                return False

class ProcessText:
    """
    This class contains method to preprocess text for training
    """

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r"\,|\<|\>|\=|[+-]?[0-9]*[,.]?[0-9]+\w+\S+\w+|[+-]?[0-9]*[,.]?[0-9]+\w+|''\w'|\w+|[^\w\s]''")
        #self.tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
        self.stopwords = set(stopwords.words('french')).union(set(stopwords.words('english')))
        self.stopwords.update({"l'","d'"})
        self.stemmer = SnowballStemmer('french')
        self.logique_operators = [">","<","="]
        self.delimiter = ","

    def tokenize_stopword_removal(self, x, text):
        word_tokens = self.tokenizer.tokenize(x)
        stop_word_removal = [w for w in word_tokens if not(w in self.stopwords)]

        if text:
            filtered_sentence = [w for w in stop_word_removal if np.logical_or.reduce((w.isupper(),w.isdigit(), w in self.logique_operators))]

        else:
            filtered_sentence = [w for w in stop_word_removal]

        return filtered_sentence

    def text_preprocess(self, x, text, snowball=True):
        """
        format text. Output tokenized words
        """

        if not pd.isnull(x):
            # differentiate acronym and normal text
            #decode_x = self.strip_accents(lower_x)
            # get rid of the space between numbers and unit
            doc = nlp(x)

            #for entity in doc.ents:
            #    print(entity.text, entity.label_)
            striped = " ".join([chunk.text for chunk in doc.noun_chunks])
            #print("Pers:", [ee for ee in doc.ents if ee.label_ == 'PERSON'])
            #print("Nouns:", [token.lemma_ for token in doc if token.pos_ == "NOUN"])
            #pdb.set_trace()
            # lower case when it's the title and when it does not correspond
            # to temperatures

            #lower_x = [w.lower() if np.logical_and(w.istitle(), not bool(re.compile(r"\d+\°\w+").search(w))) else w.upper() for w in word_tokens]

            #lower_x = [w.upper() if bool(re.compile(r"\d+\°\w+|\d+\w+").search(w)) else w.lower() for w in word_tokens]

            #stop_word_removal = [w for w in word_tokens if not(w in self.stopwords)]
            #filtered_sentence = [w for w in stop_word_removal if np.logical_or(len(w)>2,w.isupper())]

            filtered_sentence = self.tokenize_stopword_removal(striped, text)

            if snowball:
                filtered_sentence = " ".join([self.stemmer.stem(w) if not w.isupper() else w for w in filtered_sentence])
        else:
            filtered_sentence = np.NaN

        return filtered_sentence


class CrossValidation(object):
    def __init__(self,
                 classifier,
                 vectorizer,
                 params,
                 binarizer,
                 G_tags = None,
                 n_splits=5,
                 n_tags=1,
                 score_threshold=0.5,
                 l_deduplication=False,
                 l_print_errors=False):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.params = params
        self.n_splits = n_splits
        self.binarizer = binarizer
        self.G_tags = G_tags
        self.n_tags = n_tags
        self.score_threshold = score_threshold
        self.l_print_errors = l_print_errors
        self.l_deduplication = l_deduplication
        self.best_parameter_ = {}

    def _construct_pipeline(self):

        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('clf', self.classifier),])
        self.pipeline = pipeline

    def cv(self, X_train, y_train):
        kf = KFold(n_splits=self.n_splits)

        keys = list(self.params.keys())
        cv_scores_list = defaultdict(list)

        self._construct_pipeline()

        for key in keys:
            cv_scores = [[] for i in range(len(self.params[key]))]
            mean_cv_dict = dict()

            for idx, value in enumerate(self.params[key]):

                params = {key: value}


                for train_index, test_index in kf.split(X_train, y_train):

                    X_train_train = X_train.iloc[train_index]
                    y_train_train = y_train[train_index,:]

                    X_train_test = X_train.iloc[test_index]
                    y_train_test = y_train[test_index,:]

                    self.pipeline.named_steps['clf'].estimator.set_params(**params)
                    self.pipeline.fit(X_train_train, y_train_train)

                    y_pred = self.pipeline.predict(X_train_test)
                    y_pred_proba = self.pipeline.predict_proba(X_train_test)
                    y_pred_new = get_best_tags(y_pred, y_pred_proba, n_tags = self.n_tags)
                    cv_scores[idx].append(model_evaluate(y_train_test, y_pred_new,
                                                   binarizer=self.binarizer,
                                                   G_tags=self.G_tags,
                                                   score_threshold=self.score_threshold,
                                                   l_print_errors=self.l_print_errors,
                                                   l_deduplication=self.l_deduplication))

                mean_cv_dict[value] = np.mean(cv_scores[idx])
            #cv_scores_list[key].append(cv_scores)

            sorted_mean_cv_dict = sorted(
                mean_cv_dict.items(), key=itemgetter(1), reverse=True)

            self.best_parameter_[key] =  {"value": sorted_mean_cv_dict[0][0],
                                          "mean": sorted_mean_cv_dict[0][1]}
