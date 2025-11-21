# modules/text_preprocess.py
from pyvi import ViTokenizer
from pyvi.ViTokenizer import tokenize
import re
from configs import teencode_dict, stop_words

def text_preprocess(text):
    text = text.lower() # chuyển sang chữ thường
    for k, v in teencode_dict.items():
        text = re.sub(rf"\b{k}\b", v, text)
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
                  r"ùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text) # bỏ ký tự đặc biệt, dấu câu
    #text = re.sub(r"\b(" + "|".join(map(re.escape, stop_words)) + r")\b", " ", text) # loại bỏ stopword trước khi tokenize
    text = re.sub(r"\s+", " ", text).strip() # chuẩn hóa khoảng trắng 
    
    tokens = ViTokenizer.tokenize(text).split() # tokenize - tách từ
    tokens = [t for t in tokens if t not in stop_words] # loại bỏ stopword sau khi tokenize: 
    clean_text = " ".join(tokens)
    return clean_text

