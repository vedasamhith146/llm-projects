import pickle
with open("merges.pkl",'rb') as f:
    merges=pickle.load(f)
def decode(str):
    while True:
        if max(str)<256:
            break
        new_str=[]
        i=0
        reverse_merges={v:k for k,v in merges.items()}
        while i<len(str):
            if str[i] in reverse_merges:
                new_str.append((reverse_merges[str[i]])[0])
                new_str.append((reverse_merges[str[i]])[1])
                i+=1
            else :
                new_str.append(str[i])
                i+=1
        str=new_str.copy()
    str=bytes(str).decode('utf-8')
    return str