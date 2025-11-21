# ---------------------- đọc dữ liệu stopword và teencode ----------------------
STOP_WORD_FILE = './files/vietnamese-stopwords.txt'
TEENCODE_FILE = './files/teencode.txt'

stop_words =[]
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    for line in file:
        word = line.strip() # loại bỏ khoảng trắng và ký tự xuống dòng
    if word:
           stop_words.append(word)


# đọc file teen code và tạo teencode_dict {teencode: từ đầy đủ)
teencode_dict={}

with open(TEENCODE_FILE, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip() # bỏ dòng trống hoặc xuống dòng
        if not line:
            continue

        # tách theo tab
        parts = line.split("\t")
        if len(parts) ==2:
            slang, normal = parts
            teencode_dict[slang.strip()] = normal.strip()