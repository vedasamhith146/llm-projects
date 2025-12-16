from bpe_encode import encode
from bpe_decode import decode
with open("data/test.txt",'r') as f:
    file=f.read()
if decode(encode(file))==file:
    print("yayyyy you did it")
else :
    print("hmm..better luck next time")