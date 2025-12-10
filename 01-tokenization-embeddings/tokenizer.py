with open("data/tiny.txt",'r') as file:
    text=file.read()
text=list(text.encode('UTF-8'))
vocab_size=256
vocab_max=vocab_size-1
while True:
    len_text=len(text)
    print("length of text is",len_text)
    vocab={}
    for x,y in zip(text,text[1:]):
        key=(x,y)
        vocab[key]=vocab.get(key,0)+1
    if len(vocab)==0:
        break
    P=list(vocab.items())
    P.sort(key= lambda p:p[1], reverse=True)
    if P[0][1]<2:
        break
    (p1,p2)=(P[0])[0]
    vocab_size+=1
    vocab_max+=1
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
    text=new_text.copy()
print("final length of text is",len(text))


    





    

    
    
