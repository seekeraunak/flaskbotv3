"""
Routes and views for the flask application.
"""

import logging
from pprint import pprint
from collections import defaultdict
from gensim import corpora, models, similarities
#from pattern.en import mood, modality, wordnet
from os import listdir
from os.path import isfile, join
from textblob import Word
from textblob import TextBlob
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer



porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
Regextokenizer = RegexpTokenizer(r'\w+')

airbus_docs_path = 'FlaskWebProject1\\static\\airbusdocs\\'
airbus_ans_path = 'FlaskWebProject1\\static\\airbusanswers\\'

docLabels = []

docLabels = [f for f in listdir(airbus_docs_path) if f.endswith('.txt')]

documents = []
tempdocuments = []

for doc in docLabels:
    with open(airbus_docs_path + doc, 'r') as myfile:
        data1=myfile.read().replace('\n', ' ')
        data = data1.replace('.', '')
        tempdocuments.append(data)


ansdocLabels = []

ansdocLabels = [f for f in listdir(airbus_ans_path) if f.endswith('.txt')]

ansdocuments = []

for ansdoc in ansdocLabels:
    with open(airbus_ans_path + ansdoc, 'r', encoding='utf8') as ansmyfile:
        ansdata1=ansmyfile.read().replace('\n', ' ')
        ansdocuments.append(ansdata1)


for eachdoc in tempdocuments:
    tempstr = ""
    text = (Regextokenizer.tokenize(eachdoc.lower()))
    tagged_words = nltk.pos_tag(text)
    #print(tagged_words)
    #print("\n")

    for i in tagged_words:
        
        w = Word(i[0].lower())
        
        if i[1].startswith("VB"):
            lemmatized_w = wordnet_lemmatizer.lemmatize(w, 'v')

        else:
            lemmatized_w = wordnet_lemmatizer.lemmatize(w)
            
        tempstr += lemmatized_w + " "

    documents.append(tempstr.strip())



stoplist = set('compatible can for a of the and to in is are am an I i be been but do if into up my myself no or out very you your will what why how explain shall on do does'.split())
contextstoplist = set('non airbus aircraft customer compatible can for a of the and to in is are am an I i be been but do if into up my myself no or out very you your will what why how explain shall on do does'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]for document in documents]
# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
        
#texts = [[token for token in text if frequency[token] > 1] for text in texts]
#pprint(texts)


dictionary = corpora.Dictionary(texts)


corpus = [dictionary.doc2bow(text) for text in texts]


tfidf = models.TfidfModel(corpus)

#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)
lsi = models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=10)

#index = similarities.MatrixSimilarity(lsi[corpus]) #transform corpus to LSI and index it

#print(corpus)

@app.route('/_analyse')
def _analyse():
    return_message = ""
    
    doc = request.args.get('usrinp')

    if doc.lower() == "yes":
        session.clear()
        return jsonify(result = "Happy to help you!! You can ask more questions!")

    elif doc.lower() == "escalate":
        session.clear()
        return jsonify(result = "The answer to your query is not available. Your issue has been escalated. You can ask another question.")
        
    elif doc.lower() == "1" or doc.lower() == "2" or doc.lower() == "3":

        finalanswerindex = session['finalansindex']

        if doc.lower() == "1":
            ansindex = finalanswerindex[0]

        if doc.lower() == "2":
            ansindex = finalanswerindex[1]

        if doc.lower() == "3":
            ansindex = finalanswerindex[2]

        return_message += ansdocuments[ansindex]

        return jsonify(result = return_message)

    else:
        #query_text_list = [word for word in doc.lower().split() if word not in stoplist]

        query_text_list = (Regextokenizer.tokenize(doc.lower()))
        query_tagged_words = nltk.pos_tag(query_text_list)

        query_text = ""

        for i in query_tagged_words:
            query_w = Word(i[0].lower())

            if i[1].startswith("VB"):
                query_lemmatized_w = wordnet_lemmatizer.lemmatize(query_w, 'v')

            else:
                query_lemmatized_w = wordnet_lemmatizer.lemmatize(query_w)
            
            query_text += query_lemmatized_w + " "

        #return jsonify(result = query_text)

        
        vec_bow = dictionary.doc2bow(query_text.lower().split())
        vec_lsi = lsi[vec_bow] # convert the query to LSI
        #print(vec_lsi)

        index = similarities.MatrixSimilarity(lsi[corpus])

        sims = index[vec_lsi] # perform a similarity check
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        #print(sims) # print sorted (document number, similarity score)

    

        match_count = []
        match_index = -1

        for tdoc in documents:
            match_count.append(0)

        topsims = sims[:10]
        top3sim_index = []

        for temptuple in topsims[:3]:
            top3sim_index.append(temptuple[0])

        
            


        for temptuple in topsims:
            #print("--------xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-----------")
            match_index = temptuple[0]
            curr_doc = documents[temptuple[0]]
            #print(curr_doc)
            curr_doc_text = [word for word in curr_doc.lower().split() if word not in stoplist]
            #print("curr_doc_text : ", curr_doc_text)

            commonlist = list(set(curr_doc_text).intersection(set(query_text)))

            #print("commonlist : ", commonlist)

            match_count[match_index] = len(commonlist)

            #print("Match_count = ",match_count[match_index])



        top_index = sorted(range(len(match_count)), key=lambda i: match_count[i])[-3:]

        #retstmt = "<br>" + str(top_index[0]) + str(top_index[1]) + str(top_index[2])
        #return jsonify(result=retstmt)
    

        final_ans_index = []
        simcnt = 0

        retstmt = "Hi"

        for j in reversed(top_index):
            if match_count[j] == 0:
                final_ans_index.append(top3sim_index[simcnt])
                simcnt += 1

            else:
                final_ans_index.append(j)
                


        session.clear()
        session['finalansindex'] = final_ans_index

        #retstmt += "<br>" + str(final_ans_index[0]) + str(final_ans_index[1]) + str(final_ans_index[2])
        #return jsonify(result=retstmt)


        all_important_words = []
                

        #print("\nTOP 3 SIMILAR QUESTIONS : \n")
        for j in final_ans_index:
            """
            text = (Regextokenizer.tokenize(ansdocuments[j]))
            tagged_words = nltk.pos_tag(text)

            foreign_words = []
            nouns = []
            possessive_nouns = []
            #return jsonify(result="hi")

            for i in tagged_words:
                if i[1].startswith("FW"):
                    foreign_words.append(i[0].strip())

                if i[1].startswith("NN"):
                    nouns.append(i[0].strip())

                if i[1].startswith("POS"):
                    possessive_nouns.append(i[0].strip())                                

            #all_important_words.append(foreign_words + nouns + possessive_nouns)
            #return jsonify(result="hi")
            """

            blob = TextBlob(ansdocuments[j])
            nounslist = blob.noun_phrases
            nounlength = [len(sentence.split()) for sentence in nounslist]
            top_noun_index = sorted(range(len(nounlength)), key=lambda i: nounlength[i])[-2:]
            #return jsonify(result="hi")

            topnounlist = []

            if len(top_noun_index) > 1:
                one = top_noun_index[-1]
                two = top_noun_index[-2]    
                #return jsonify(result="hi")

                if one == two:
                    topnounlist.append(nounslist[one])
            
                else:
                    topnounlist.append(nounslist[one])
                    topnounlist.append(nounslist[two])
            
            #longest_context = max(blob.noun_phrases, key=len)
            all_important_words.append(topnounlist)

        

        str1 = "Select the context (1/2/3) : <br>"

        counter = 0

        for i in range(0,len(all_important_words)):
            #all_important_words[i] = [word.lower() for word in all_important_words[i] if word.lower() not in contextstoplist]
            if len(all_important_words[i]):
                counter += 1
                str1 += str(counter) + ' : '
                for j in range(0,len(all_important_words[i])):
                    str1 += all_important_words[i][j] + ' , '

                str1 = str1.strip(' ,')

                str1 += "<br>"

            
        return jsonify(result=str1)

@app.route('/')
def index():        
    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = 'RAUNAK123'
    app.run(debug = True)
