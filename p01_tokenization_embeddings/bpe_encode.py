import pickle
with open ("merges.pkl",'rb') as f:
    merges=pickle.load(f)
def encode(text):
    text=list(text.encode('utf-8'))
    while True:
        i=c=0
        new_text=[]
        while i<len(text)-1:
            if (text[i],text[i+1]) in merges:
                new_text.append(merges[(text[i],text[i+1])])
                i+=2
                c+=1
            else :
                new_text.append(text[i])
                i+=1
        if i!=len(text):
            new_text.append(text[i])
        text=new_text.copy()
        if c==0:
            break
    return text

