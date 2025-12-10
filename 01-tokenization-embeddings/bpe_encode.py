import pickle
with ("merges.pkl",'rb') as f:
    merges=pickle.load(f)
def encode(str):
    str=list(str.encode('utf-8'))
    i=0
    new_str=[]
    if (str[i],str[i+1]) in merges:
        new_str.append(merges[(str[i],str[i+1])])
        i+=2
    else :
        new_str.append(str[i])
        i+=1
    if i!=len(str):
        new_str.append(str[i])
    str=new_str.copy()
    return str

