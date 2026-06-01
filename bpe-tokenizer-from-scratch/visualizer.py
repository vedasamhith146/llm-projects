from bpe_decode import decode
from bpe_encode import encode

def visualizer(text,merges_file):
    print(f"Original text : {text}")
    text=encode(text,merges_file)
    for i in range(len(text)):
        print(f"[{text[i]}]:{decode([text[i]],merges_file)}")
if __name__=="__main__":
    text="hello world"
    visualizer(text,merges_file="merges_100.pkl")