from bpe_decode import decode
from bpe_encode import encode

def visualizer(text):
    text=encode(text)
    print("Text with token boundaries")
    for i in range(len(text)):
        print(f"[{text[i]}]:{decode([text[i]])}")
if __name__=="__main__":
    with open("data/testnew.txt",'r') as f:
        new=f.read()
    visualizer(new)