# modules/recommender.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from configs import teencode_dict, stop_words


from modules.text_preprocess import text_preprocess

def get_recommendations(selected_index, cosine_sim_matrix, df, top_n=5):
    # Lấy vector similarity của xe được chọn
    sim_scores = list(enumerate(cosine_sim_matrix[selected_index]))

    # Sắp xếp theo mức độ tương đồng giảm dần
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Bỏ chính nó (index 0 là nó)
    sim_scores = sim_scores[1: top_n + 1]

    # Lấy index của xe tương tự
    recommend_indices = [i[0] for i in sim_scores]

    # Trả về DataFrame kết quả
    result = df.iloc[recommend_indices]
    result['cosine_sim'] = [i[1] for i in sim_scores]

    return result


def get_text_recommendations(
        search_str: str,
        df: pd.DataFrame, 
        model,
        top_n=5):
    search_str_wt = text_preprocess(search_str)

    vectorizer = TfidfVectorizer(analyzer='word', stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(df['Content_wt'])
    query_vec = vectorizer.transform([search_str_wt])
    #Tính cosine similarity giữa string text và toàn bộ Dataframe
    searchtext_cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Lấy top K kết quả
    top_indices = searchtext_cosine_sim.argsort()[::-1][:top_n]
    # Trả về kết quả

    result = df.iloc[top_indices].copy()
    result['cosine_similarity'] = searchtext_cosine_sim[top_indices]


    return result