import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modules.loader import load_data, load_model, load_data_cluster
from modules.recommender import get_recommendations, get_text_recommendations
from modules.text_preprocess import text_preprocess



# ---------------------- DATA DEMO ----------------------
# Dummy data cho phân cụm
np.random.seed(42)
df_demo = pd.DataFrame({
    "Giá": np.random.randint(15, 80, 80),
    "Km": np.random.randint(5000, 60000, 80),
    "Dung tích": np.random.choice([110, 125, 150], 80),
})
# Gán cụm giả lập
df_demo["cluster"] = np.random.randint(0, 3, 80)

# Dummy dataset cho content-based
motorbikes = pd.DataFrame({
    "name": ["Honda Vision", "Yamaha Sirius", "Honda AirBlade", "Yamaha Janus", "Honda Lead"],
    "brand": ["Honda", "Yamaha", "Honda", "Yamaha", "Honda"],
    "engine": [110, 110, 125, 110, 125],
    "price": [28, 20, 40, 30, 38],
    "description": [
        "Xe tay ga nhỏ gọn tiết kiệm xăng",
        "Xe số phổ thông bền bỉ giá rẻ",
        "Xe tay ga cao cấp mạnh mẽ",
        "Xe tay ga nhẹ nhàng phù hợp nữ",
        "Xe tay ga rộng rãi cốp lớn",
    ],
})

# tạo vector mô phỏng similarity
motorbikes["desc_vec"] = motorbikes.index

# ---------------------- APP ----------------------
def main():

    st.set_page_config(page_title="Motorbike Recommendation", layout="wide")

    # ---------- SIDEBAR WITH LOGO & INFO ----------
    st.sidebar.image("xe_may_cu.jpg", width=80)
    st.sidebar.title("🚀 Menu")

    menu = st.sidebar.radio(
        "Đi đến mục:",
        [
            "Business Problem",
            "Evaluation & Report",
            "Content-Based Recommendation",
            "Thông tin nhóm thực hiện"
        ]
    )

    # ---------- BUSINESS PROBLEM ----------
    if menu == "Business Problem":
        st.title("📌 Business Problem")
        st.markdown(
            """
            ### Bối cảnh
            Người mua xe máy cũ gặp nhiều khó khăn vì thị trường đa dạng, giá chênh lệch và thông tin thiếu minh bạch.

            ### Mục tiêu dự án
            - Xây dựng hệ thống gợi ý xe máy cũ phù hợp nhu cầu.
            - Sử dụng Content-Based filtering và phân cụm để đưa ra gợi ý.
            - Hiển thị phân tích dữ liệu, báo cáo hiệu suất mô hình.
            """
        )

    # ---------- EVALUATION (A + C) ----------
    elif menu == "Evaluation & Report":
        df_cluster = load_data_cluster()
        X = df_cluster[['Giá', 'Số Km đã đi', 'Dung tích xe_encoded', 'Năm đăng ký']].dropna()
        # chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.title("📊 Evaluation & Clustering Report")

        st.subheader("Biểu đồ Phân Cụm (PCA 2D)")

        # PCA 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_cluster["pca1"] = X_pca[:, 0]
        df_cluster["pca2"] = X_pca[:, 1]

        centroids = df_cluster.groupby('agg_cluster')[['pca1', 'pca2']].mean()
        # centroids


        fig, ax = plt.subplots()
        scatter = ax.scatter(df_cluster["pca1"], df_cluster["pca2"], c=df_cluster["cluster"], alpha = 0.3)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

        # Ghi nhãn centroid
        for idx, row in centroids.iterrows():
            ax.text(row['pca1'], row['pca2'], 'x', fontsize=12, weight='bold')

        st.pyplot(fig)

        # --- C. Mô tả cụm ---
        st.subheader("📌 Mô tả các cụm (Cluster Summary)")
        st.markdown(
            """
            **Cụm 0:** Xe có mức giá nhỏ hơn 100tr, quãng đường đã đi < ~300km, dung tích 175cc, phù hợp với nhu cầu thông thường. 
            **Cụm 1:** Xe giá rẻ, đã đi rất nhiều.  
            **Cụm 2:** Xe phân khối lớn và giá cao.  
            """
        )

        st.subheader("📈 Phân bố giá theo từng cụm")
        fig2, ax2 = plt.subplots(figsize=(3, 2))
        for cl in df_cluster["cluster"].unique():
            ax2.hist(df_cluster[df_cluster.cluster == cl]["Giá"], alpha=0.5, label=f"Cluster {cl}")
        ax2.legend()
        st.pyplot(fig2)

        st.image("kmcluster.png")
        st.image("namdungtichcluster.png")

    # ---------- RECOMMENDATION (B) ----------
    elif menu == "Content-Based Recommendation":
        st.title("🔍 Gợi ý theo Content-Based Filtering")

        # --- LOAD DỮ LIỆU ---
        df = load_data()
        df_cluster = load_data_cluster()
        model = load_model()  # cosine similarity matrix
        # df = pd.read_csv("cho_tot_cleaned_wt.csv")
       

        # st.subheader("📋 Danh sách xe máy")
        # st.write("Chọn một xe để xem thông tin và gợi ý:")

        # ===========================
        #  Chọn chế độ
        # ===========================
        mode = st.radio(
            "Chọn phương thức gợi ý:",
            ["Chọn từ danh sách", "Tìm theo nội dung nhập vào"]
        )

        # ===========================
        #  MODE 1: CHỌN TỪ DANH SÁCH
        # ===========================
        if mode == "Chọn từ danh sách":
            st.markdown("Lọc danh sách xe theo thương hiệu và mức giá:")
            
            brand_list = ["Tất cả"] + sorted(df['Thương hiệu'].dropna().unique().tolist())
            brand = st.selectbox("Thương hiệu mong muốn", brand_list)

            price_range = st.slider("Khoảng giá (triệu)", 15, 200, (20, 50))
                          
            st.subheader("📋 Chọn xe từ danh sách")
            # Danh sách tiêu đề xe để người dùng chọn
            if brand != "Tất cả":
                df_filtered = df[df["Thương hiệu"] == brand]
            else: df_filtered = df
            # Filter theo giá
            df_filtered = df_filtered[(df_filtered['Giá']>= price_range[0]) & (df_filtered['Giá'] <= price_range[1])]
            xe_list = df_filtered['Tiêu đề'].tolist()
            if len(xe_list) > 0:

                selected_xe = st.selectbox("Chọn một xe để xem thông tin và gợi ý tương tự", xe_list)

                # Tìm index của xe được chọn
                selected_index = df.index[df['Tiêu đề'] == selected_xe][0]

                # # Lấy thông tin xe đã chọn
                selected_row = df.loc[selected_index]

                # Tìm xe tương ứng trong df_cluster
                matched = df_cluster[df_cluster['Tiêu đề'] == selected_xe]
                if len(matched) > 0:
                    cluster_value = matched.iloc[0]['agg_cluster']
                    st.success(f"🚗 Xe này thuộc **cụm {cluster_value}**")
                else:
                    st.warning("⚠ Xe này **không có cụm tương ứng** trong dữ liệu phân cụm.")

                st.write("### **🔍 Thông tin xe đã chọn:**")
                st.json(selected_row.to_dict())
                #st.dataframe(df.iloc[[selected_index]])
                # =====================================
                # Gọi model để tìm xe tương tự
                # =====================================
                # Lấy gợi ý từ model
                recommendations = get_recommendations(selected_index, model, df, top_n=5)

                st.write("### 🔎 Gợi ý xe tương tự:")
                st.dataframe(recommendations)
            else: 
                st.warning("❗ Không có xe phù hợp với thương hiệu và mức giá đã chọn.")



        # ===========================
        #  MODE 2: NHẬP NỘI DUNG
        # ===========================
        else:
            st.subheader("✏️ Nhập nội dung để tìm xe phù hợp")

            user_input = st.text_area("Nhập mô tả (vd: xe tay ga, vespa, chính chủ...)", value= "vespa sprint chính chủ")

            if st.button("Tìm kiếm"):
                if len(user_input) < 3:
                    st.error("Vui lòng nhập nội dung đủ dài.")
                else:
                    # =====================================
                    # Model xử lý text và trả kết quả
                    # =====================================

                 
                    #user_input_wt = text_preprocess(user_input)
                    # vectorizer = TfidfVectorizer(analyzer='word', stop_words=stop_words)
                    # query_vec = vectorizer.transform([search_str_wt])
                    # #Tính cosine similarity giữa string text và toàn bộ Dataframe
                    # searchtext_cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
                    # top5_idx = searchtext_cosine_sim.argsort()[::-1][:5]
                    # df[['id', 'Giá', 'Dòng xe','Năm đăng ký','Loại xe','Dung tích xe', "Mô tả chi tiết"]].iloc[top5_idx]
                    # df_results = df.iloc[top5_idx].copy()
                    # df_results['cosine_similarity'] = searchtext_cosine_sim[top5_idx]

                    str_recommendations = get_text_recommendations(user_input, df, model, top_n=5)

                    st.write("### 🔎 Kết quả gợi ý:")

                    st.dataframe(str_recommendations)


        
        #st.dataframe(recommendations[["id", "title", "description"]])




        # st.title("🔍 Recommendation Content-Based")

        # st.markdown("Nhập nhu cầu để gợi ý xe phù hợp.")

        # brand = st.selectbox("Thương hiệu mong muốn", ["Honda", "Yamaha", "Không quan trọng"])
        # price_range = st.slider("Khoảng giá (triệu)", 15, 80, (20, 50))
        # keyword = st.text_input("Từ khoá mô tả (ví dụ: tiết kiệm, mạnh mẽ, nhẹ nhàng)")

            

        # if st.button("Gợi ý ngay"):
        #     df = motorbikes.copy()

        #     # Filter theo giá
        #     df = df[(df.price >= price_range[0]) & (df.price <= price_range[1])]

        #     # Filter theo brand
        #     if brand != "Không quan trọng":
        #         df = df[df.brand == brand]

        #     # Mô phỏng cosine similarity
        #     if keyword:
        #         scores = []
        #         for desc in df.description:
        #             sim = len(set(keyword.split()) & set(desc.split()))  # mô phỏng đơn giản
        #             scores.append(sim)
        #         df["similarity"] = scores
        #         df = df.sort_values("similarity", ascending=False)

        #     st.subheader("✨ Top Xe Gợi Ý")
        #     st.table(df[["name", "brand", "engine", "price"]])
        
        

    # ---------- TEAM INFO (D) ----------
    else:
        st.title("👥 Thông tin nhóm thực hiện")

        st.markdown(
            """
            ### Nhóm dự án Recommendation Motorbike
            - Nguyễn Ngọc Giao – GUI Project 1
            - Nguyễn Thị Tuyển – GUI Project 2
        
            ### Liên hệ
            📧 Email: group@example.com  
            💻 Github: https://github.com/group
            """
        )


if __name__ == "__main__":
    main()
