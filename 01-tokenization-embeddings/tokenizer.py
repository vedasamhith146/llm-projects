with open("data/tiny.txt",'r') as file:
    text=file.read()
text=list(text.encode('UTF-8'))
len_text=len(text)
vocab_size=256
vocab_max=vocab_size-1
for x in range(10):
    vocab={}
    for x,y in zip(text,text[1:]):
        key=(x,y)
        vocab[key]=vocab.get(key,0)+1
    P=list(vocab.items())
    P.sort(key= lambda p:p[1], reverse=True)
    (p1,p2)=(P[0])[0]
    print("the highest frequency is of",(P[0])[0],"and it's value is ",(P[0])[1])
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


    





    

    
    
