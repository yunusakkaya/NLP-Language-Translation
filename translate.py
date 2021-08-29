#%%word2vec
"""import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

f=open('h.txt','r',encoding='utf8')
text=f.read()
t_list=text.split('\n')

corpus=[]
for  cumle in t_list:
    corpus.append(cumle.split())
    
model= Word2Vec(corpus, size=100, window=5, min_count=5)

model.wv['ankara']
model.wv.most_similar('hollanda')

model.save('word2vec.model')
model= Word2Vec.load('word2vec.model')

def closestwords_tsneplot(model,word):
    word_vectors=np.empty((0,100))
    word_labels=(word)
    
    close_words= model.wv.most_similar(word)
    word_vectors.append(word_vectors,np.array([model.wv[word]]),axis=0)
    
    for w,_in close_words:
        word_labels.append(w)
        word_vectors=np.append(word_vectors,np.array([model.wv[w]]),axis=0)
    
    tsne=TSNE(random_state=0)
    Y=tsne.fit_transform(word_vectors)
    
    x_coords=Y[:,0]
    y_coords=Y[:,1]
    
    plt.scatter(x_coords,y_coords)
    
    for label,x,y in zip(word_labels,x_coords,y_coords):
        plt.annotate(label,xy=(x,y),xytext=(5,-2),textcoords='offset points')
        plt.show()"""
        
#%%glove 
"""from gensim.scripts.glove2word2vec import glove2word2vec
glove_input='glove.6B.100d.txt'
word2vec_output='glove.6B.100d.word2vec'
glove2word2vec(glove_input,word2vec_output)

model=KeyedVectors.load_word2vec_format(word2vec_output,binary=False)
model['İstanbul']"""

#%%SA
"""import numpy as np
import pandas as pd
from tenserflow.python.keras.model import Sequential
from tenserflow.python.keras.layers import Dense,GRU,Embedding,CuDNNGRU
from tenserflow.python.keras.optimizers import Adam
from tenserflow.python.keras.preprocessin.text import Tokenizer
from tenserflow.python.keras.preprocessin.sequence import pad_sequences

dataset=pd.read_csv(r'hepsiburada.csv')

target=dataset['Rating'].values.tolist()
data=dataset['Review'].values.tolist()

cutoff=int(len(data)*0.80)
x_train,x_test=data[:cutoff],data[cutoff:]
y_train,y_test=target[:cutoff],target[cutoff:]

num_words=10000
tokenizer=Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data)

x_train_tokens=tokenizer.texts_to_sequences(x_train)
x_test_tokens=tokenizer.texts_to_sequences(x_test)

num_tokens=[len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens=np.array(num_tokens)
max_tokens=np.mean(num_tokens)+2*np.std(num_tokens)
max_tokens=int(max_tokens)
np.sum(num_tokens<max_tokens)/len(num_tokens)

x_train_pad=pad_sequence(x_train_tokens,maxlen=max_tokens)
x_test_pad=pad_sequence(x_test_tokens,maxlen=max_tokens)

idx=tokenizer.word_index
inverse_map=dict(zip(idx.values(),idx.keys()))
def tokens_to_string(tokens):
    words=[inverse_map[token] for token in tokens if token!=0 ]
    text=''.join(words)
    return text

model=Sequential()
embedding_size=50
model.add(Embedding(input_dim=num_words,output_dim=embedding_size,input_lenght=max_tokens,name='embedding_layer'))
model.add(GRU(units=16,return_sequences=True))
model.add(GRU(units=8,return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1,activation='sigmoid'))

optimizer=Adam(1r=1e-3)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()
model.fit(x_train_pad,y_train,epochs=5,batch_size=256)

result=model.evalaute(x_test_pad,y_test)
result[1]

y_pred=model.predict(x=x_test_pad[0:1000])
y_pred=y_pred.T[0]

cls_pred=np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
cls_true=np.array(y_test[0:1000])
incorrect=np.where(cls_pred!=cls_true)
incorrect=incorrect[0]
len(incorrect)
idx=incorrect[0]
idx
text=x_test[idx]
text
y_pred[idx]
cls_true[idx]"""

#%% NMT
"""import numpy as np
import tensorflow as tf
from tenserflow.python.keras.model import Sequential
from tenserflow.python.keras.layers import Dense,GRU,Embedding,CuDNNGRU
from tenserflow.python.keras.optimizers import Adam
from tenserflow.python.keras.preprocessin.text import Tokenizer
from tenserflow.python.keras.preprocessin.sequence import pad_sequences

mark_started='ssss'
mark_end='eeee'

data_src=[]
data_dest=[]

for line in open('tur.txt',encoding='UTF-8'):
    en_text,tr_text=line.rstrip().split('\t')
    
    tr_text=mark_start + tr_text + mark_end
    data_src.append(en_text)
    data_dest.append(tr_text)
    
class TokenizerWrap(Tokenizer):
    def __init__(self,texts,padding,reverse=False,num_words=None):
        Tokenizer.__init__(self,num_words=num_words)
        
        self.fit_on_texts(texts)
        self.index_to_word=dict(zip(self.word_index.values(), self.word_index.keys()))
        self.tokens= self.texts_to_sequences(texts)
        
        if reverse:
            self.tokens=[list(reversed(x)) for x in self.tokens]
            truncating='pre'
        else:
            truncating='post'
        
        self.num_tokens=[len(x) for x in self.tokens]
        self.max_tokens=np.mean(self.num_tokens) + 2*np.std(self.num_tokens)
        self.max_tokens=int(self.max_tokens)
        
        self.tokens_padded=pad_sequences(self.tokens,maxlen=self.max_tokens,padding=padding,truncating=truncating)
        
    def token_to_word(self,token):
        word=''if token ==0 else self.index_to_word[token]
        return word
    
    def tokens_to_string(self,tokens):
        word=[self.index_to_word[token] for token in tokens if token != 0]
        text= ''.join(words)
        return text
    
    def text_to_tokens(self,text,padding,reverse=False):
        tokens=self.text_to_sequences([text])
        tokens=np.array(tokens)
        
        if reverse:
            tokens=np.flip(tokens,axis=1)
            truncating='pre'
        else:
            truncating='post'
            
        tokens=pad_sequences(tokens,maxlen=self.max_tokens,padding=padding,truncating=truncating)
        return tokens
    
tokenizer_src=TokenizerWrap(text=data_src,padding='pre',reverse=True,num_words=None)
tokenizer_dest=TokenizerWrap(text=data_dest,padding='post',reverse=True,num_words=None)    
tokens_src=tokenizer_src.tokens_padded            
tokens_dest=tokenizer_dest.tokens_padded            
print(tokens_src.shape)            
print(tokens_dest.shape)

token_start=tokenizer_dest.word_index[mark_start.strip()]
token_end=tokenizer_dest.word_index[mark_end.strip()]            
            
encoder_input_data=tokens_src

decoder_input_data=tokens_dest[:, :-1]
decoder_output_data=tokens_dest[:, :1]

num_encoder_words=len(tokenizer_src.word_index)
num_decoder_words=len(tokenizer_dest.word_index)

embedding_size=100
word2vec={}
with open('glove.6B.100d.txt', encoding='UTF-8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vec=np.asarray(values[1:],dtype='float32')
        word2vec[word]=vec
        
embedding_matrix=np.random.uniform(-1,1,(num_encoder_words,embedding_size))
for word, i in tokenizer_src.word_index.items():
    if i<num_encoder_words:
        embedding_vector=word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
            
print(embedding_matrix.shape)

encoder_input= Input(shape=(None,),name='encoder_input')
encoder_embedding=Embedding(input_dim=encoder_words,output_dim=embedding_size,weights=[embedding_matrix],trainable=True,name='encoder_embedding')

state_size=256

encoder_gru1=GRU(state_size,name='encoder_gru1', return_sequences=True)        
encoder_gru2=GRU(state_size,name='encoder_gru2', return_sequences=True)             
encoder_gru3=GRU(state_size,name='encoder_gru3', return_sequences=False)

def connect_encoder():
    net= encoder_input
    net= encoder_embedding(net)
    net= encoder_gru1(net)
    net= encoder_gru2(net)        
    net= encoder_gru3(net)
    encoder_output= net
    return encoder_output

encoder_output= connect_encoder()

decoder_initial_satate=Input(shape=(state_size,),name='decoder_initial_state')
decoder_input=Input(shape=(None,),name='decoder_input')
decoder_embedding=Embedding(input_dim=num_decoder_words,output_dim=embedding_size,name='decoder_embedding')
decoder_gru1=GRU(state_size,name='decoder_gru1',return_sequences=True)
decoder_gru2=GRU(state_size,name='decoder_gru2',return_sequences=True)
decoder_gru3=GRU(state_size,name='decoder_gru3',return_sequences=True)
decoder_dense=Dense(num_decoder_words,activation='linear',name='decoder_output')

def connect_decoder(initial_state):
    net=decoder_input
    net=decoder_embedding(net)
    net=decoder_gru1(net,initial_state=initial_state)
    net=decoder_gru2(net,initial_state=initial_state)
    net=decoder_gru3(net,initial_state=initial_state)
    decoder_output=decoder_dense(net)
    return decoder_output

decoder_output=connect_decoder(initial_state=encoder_output)
model_train= Model(inputs=[encoder_input,decoder_input],outputs=[decoder_output])
model_encoder= Model(inputs=[encoder_input],outputs=[encoder_output])

decoder_output=connect_decoder(initial_state=decoder_initial_state)
model_decoder= Model(inputs=[decoder_input,decoder_initial_state],outputs=[decoder_output])



def sparse_cross_entropy(y_true,y_pred):
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss_mean=tf.reduce_mean(loss)
    return loss_mean


optimizer_target=tf.placeholder(dtype='int32',shape=(None,None))
model_train.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])
path_checkpoint='checkpoint.keras'
checkpoint= ModelCheckpoint(filepath=path_checkpoint,save_weights_only=True)

try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print(error)
    
x_data={'encoder_input':encoder_input_data, 'decoder_input':decoder_input_data}
y_data={'decoder_output':decoder_output_data}
    
model_train.fit(x=x_data,y=y_data,batch_size=256,epochs=2,callbacks=[checkpoint])    



def translate(input_text, true_output_text=None):
    input_tokens= tokenizer_src.text_to_tokens(text=input_text,reverse=True,padding='pre')
    initial_state=model_encoder.predict(input_tokens)
    
    max_tokens=tokenizer_dest.max_tokens
    decoder_input_data=np.zeros(shape=(1,max_tokens),dtype=np.int)
    
    token_int=token_start
    output_text=''
    count_tokens=0
    
    while token_int != token_end and count_tokens< max_tokens:
        decoder_input_data[0,count_tokens] = token_int
        x_data={'decoder_initial_state':initial_state,'decoder_input':decoder_input_data}
        
        decoder_output= model_decoder.predict(x_data)
        
        token_onehot= decoder_output[0,count_tokens,:]
        token_int= np.argmax(token_onehot)
        sampled_word= tokenizer_dest.token_to_word(token_int)
        output_text += ''+sampled_word
        output_tokens += 1
        
    print('Input_text:')
    print(input_text)    
    print()
    print('translated text')
    print(output_text)
    print()
    
    if true_output_text is not None:
        print("true output text:", true_output_text)
        
  
translate(input_text='This summer I went to istanbul')"""



#%% İMCAP
import tensorflow as tf
from tenserflow.python.keras.model import Sequential
from tenserflow.python.keras.layers import Dense,GRU,Embedding,CuDNNGRU
from tenserflow.python.keras.optimizers import Adam
from tenserflow.python.keras.preprocessin.text import Tokenizer
from tenserflow.python.keras.preprocessin.sequence import pad_sequences
from PIL import image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import coco
from coco import cache

coco.maybe_dowland_and_extract()
_,filenames,captions= coco.load_records(train=True)

num_images=len(filenames)

def load_image(path,size=None):
    img=Image.open(path)
    
    if not size is None:
        img=img.resize(size=size,resample=Image.LANCZOS)
    return img
    
def show_image(idx):
    dir=coco.train_dir
    filename= filename[idx]
    caption= caption[idx]
    path=os.path.join(dir,filename)
    
    for cap in caption:
        print(cap)
        img=load_image(path)
        plt.imshow(img)
        plt.show
                       
show_image(idx=1)     

image_model = VGG16()
image_model.summary()

transfer_layer= image_model.get_layer('fc2')
image_model_transfer= Model(inputs=image_model.input,outputs=transfer_layer.output)

img_size= K.int_shape(image_model.input)[1:3]
image_size

transfer_values_size = K.int_shape(transfer_layer.output)[1]
transfer_values_size

def print_progress(count,max_count):
    pct_complete= count/max_count
    msg= '\r-Progress: {0:1%}'.format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()
    
def process_images(data_dir,filenames,batch_size=32):
    num_images=len(filenames)
    shape= (batch_size,) + img_size + (3,)
    image_batch= np.zeros(shape=shape,dtype=np.float16)
    shape=(num_images,transfer_values_size)
    transfer_values= np.zeros(shape=shape,dtype=np.float16)
    start_index=0
    
    while start_index < num_images:
        print_progress(count= start_index,max_count=num_images)
        end_index= start_index + batch_size
        
        if end_index > num_images:
            end_index= num_images
        
        current_batch_size= end_index - start_index
        
        for i, filename in enumerate(filenames[start_index:end_index]):
            path= os.path.join(data_dir,filename)
            img= load_image(path,size=img_size)
            image_batch[i]=img
            
        transfer_values_batch= image_model_transfer.predict(image_batch[0:current_batch_size])
        
        transfer_values[start_index:end_index]=transfer_values_batch[0:current_batch_size]
        start_index=end_index
        
    return transfer_values
        
def process_train_images():
    print('{} resim işleniyor'.format(len(filenames)))
    cache_path= os.path.join(coco.data_dir,'transfer_values_train.pkl')
    transfer_values= cache(cache_path=cache_path,fn=process_images,data_dir=coco.train_dir,filenames=filenames)
    return transfer_values

# %%time
transfer_values= process_train_images()
print('Shape:',transfer_values.shape)

mark_start='ssss'
mark_end='eeee'

def mark_captions(captions_listlist):
    captions_marked=[[mark_start + caption + mark_end for caption in captions_list]
                      for captions_list in captions_listlist]
        
    return captions_marked

captions_marked=mark_captions(captions)

def flatten(captions_listlist):
    captions_list= [caption for captions_list in captions_lislist for caption in captions_list]
    return captions_list

captions_flat= flatten(captions_marked)

class TokenizerWrap(Tokenizer):
    def __init__(self,texts,num_words=None):
        Tokenizer.__init__(self,num_words=num_words)
        
        self.fit_on_texts(texts)
        self.index_to_word=dict(zip(self.word_index.values(), self.word_index.keys()))
        
    def token_to_word(self,token):
        word=''if token ==0 else self.index_to_word[token]
        return word
    
    def tokens_to_string(self,tokens):
        word=[self.index_to_word[token] for token in tokens if token != 0]
        text= ''.join(words)
        return text    
    def captions_to_tokens(self,captions_listlist):
        tokens=[self.texts_to_sequences(captions_list) for captions_list in captions_listlist]
        
        return tokens
    
tokenizer= TokenizerWrap(texts= captions_flat, num_words=num_words)
tokens_train= tokenizer.captions_to_tokens(captions_marked)

token_start= tokenizer.word_index[mark_start.strip()]        
token_end= tokenizer.word_index[mark_end.strip()]          
        
        
            
def get_random_caption_tokens(idx):
    result=[]
    
    for i in idx:
        j= np.random.choice(len(tokens_train[i]))
        tokens= tokens_train[i][j]
        result.append(tokens)
    return result

def batch_generator(batch_size):
    while True:
        idx= np.random.randint(num_images,size=batch_size)
        t_values= transfer_values[idx]
        tokens= get_random_caption_tokens(idx)
        
        num_tokens= [len(t) for t in tokens]
        max_tokens= np.max(num_tokens)
        
        tokens_padded = pad_sequences(tokens,maxlen=max_tokens,padding='post',truncating='post')
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]
        
        x_data={'decoder_input':decoder_input_data,'transfer_values_input':t_values}
        y_data={'decoder_output':decoder_output_data}
        yield (x_data,y_data)
        
batch_size=256
generator= batch_generator(batch_size)
batch=next(generator)
batch_x= batch[0]
batch_y= batch[1]



num_captions = [len(caption) for caption in captions]
total_num_captions = np.sum(num_captions)

steps_per_epoch = int(total_num_captions/batch_size)

state_size=256
embedding_size= 100
transfer_values_input=Input(shape=(transfer_values_size,),name='transfer_values_input')
decoder_transfer_map= Dense(state_size,activation='tanh',name='decoder_transfer_map')
decoder_input= Input(shape=(None,),name='decoder_input')

word2vec={}
with open('glove.6B.100d.txt', encoding='UTF-8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vec=np.asarray(values[1:],dtype='float32')
        word2vec[word]=vec
        
embedding_matrix=np.random.uniform(-1,1,(num_encoder_words,embedding_size))
for word, i in tokenizer_src.word_index.items():
    if i<num_encoder_words:
        embedding_vector=word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
            
            
decoder_embedding=Embedding(input_dim=num_decoder_words,output_dim=embedding_size,name='decoder_embedding')
decoder_gru1=GRU(state_size,name='decoder_gru1',return_sequences=True)
decoder_gru2=GRU(state_size,name='decoder_gru2',return_sequences=True)
decoder_gru3=GRU(state_size,name='decoder_gru3',return_sequences=True) 
decoder_dense=Dense(num_decoder_words,activation='linear',name='decoder_output')

def connect_decoder(initial_state):
    net=decoder_input
    net=decoder_embedding(net)
    net=decoder_gru1(net,initial_state=initial_state)
    net=decoder_gru2(net,initial_state=initial_state)
    net=decoder_gru3(net,initial_state=initial_state)
    decoder_output=decoder_dense(net)
    return decoder_output
      
decoder_output= connect_decoder(transfer_values=transfer_values_input)
decoder_model=Model(inputs=[transfer_values_input,decoder_input],outputs=[decoder_output])

def sparse_cross_entropy(y_true,y_pred):
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss_mean=tf.reduce_mean(loss)
    return loss_mean

optimizer=RMSprop(1r=1e-3)
decoder_target=tf.placeholder(dtype='int32',shape=(None,None))
decoder_model.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])

path_checkpoint='checkpoint.keras'
checkpoint= ModelCheckpoint(filepath=path_checkpoint,save_weights_only=True)

try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print(error)
    
decoder_model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch,epoch=20,callbacks=[checkpoint])


def generate_caption(image_path,max_tokens=30):
    image=load_image(image_path,size=img_size)
    image_batch= np.expand_dims(image,axis=0)
    transfer_values= image_model_transfer.predict(image_batch)
    
    while token_int != token_end and count_tokens< max_tokens:
        decoder_input_data[0,count_tokens] = token_int
        x_data={'decoder_initial_state':initial_state,'decoder_input':decoder_input_data}
        
        decoder_output= model_decoder.predict(x_data)
        
        token_onehot= decoder_output[0,count_tokens,:]
        token_int= np.argmax(token_onehot)
        sampled_word= tokenizer_dest.token_to_word(token_int)
        output_text += ''+sampled_word
        output_tokens += 1
        
    plt.imshow(image)
    plt.show()
    
    print('predicted caption:',output_text)
    
def generate_caption_coco(idx):
    data_dir=coco_train_dir
    filename=filename[idx]
    caption= captions[idx]
    
    path= os.path.join(data_dir,filename)
    generate_caption(image_path=path)
    
    print('true captions:')
    for cap in caption:
        print(cap)    
            
generate_caption_coco(idx=1)    

      
      






















































        
        

 
