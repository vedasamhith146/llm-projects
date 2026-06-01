import pickle

def train_tokenizer(input_file, max_merges, min_freq):

    with open(input_file,'r') as file:
        text=file.read()
    text=list(text.encode('utf-8'))                           #encode function returns byte format b'' 

    vocab_size=256
    vocab_max=vocab_size-1
    my_generated_merges={}
    merges=0

    while merges<max_merges:
        len_text=len(text)
        vocab={}
        for x,y in zip(text,text[1:]):
            key=(x,y)
            vocab[key]=vocab.get(key,0)+1                                      
        P=list(vocab.items())
        P.sort(key= lambda p:p[1], reverse=True)
        freq=P[0][1]
        if freq>=min_freq:
            (p1,p2)=(P[0])[0]
            vocab_size+=1
            vocab_max+=1
            my_generated_merges[(p1,p2)]=vocab_max
            merges+=1
            i=0
            new_text=[]
            while i<len_text-1:
                if (text[i]==p1) and (text[i+1]==p2) :
                    new_text.append(vocab_max)
                    i+=2
                else :
                    new_text.append(text[i])
                    i+=1
            if i!=len_text:
                new_text.append(text[i])
        else:
            break
        text=new_text.copy()
    with open(f"merges_{max_merges}.pkl","wb") as f:
        pickle.dump(my_generated_merges,f)
    

if __name__=="__main__":
    input_file="data/tiny.txt"
    train_tokenizer(input_file=input_file,max_merges=2500,min_freq=2)

        
