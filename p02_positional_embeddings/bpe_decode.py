import pickle
with open("merges.pkl",'rb') as f:
    merges=pickle.load(f)
def decode(text):
    reverse_merges={v:k for k,v in merges.items()} 
    while any(p in reverse_merges for p in text) :
        new_text=[]
        i=0
        while i<len(text):
            if text[i] in reverse_merges:
                new_text.append((reverse_merges[text[i]])[0])
                new_text.append((reverse_merges[text[i]])[1])
                i+=1
            else :
                new_text.append(text[i])
                i+=1
        text=new_text.copy()
    text=bytes(text).decode('utf-8',errors="replace")
    return text