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
    #st.title("Trung Tâm Tin Học")
    st.image("xe_may_cu.png", caption="Chợ mua bán xe máy cũ")
    # ---------- SIDEBAR WITH LOGO & INFO ----------
    #st.sidebar.image("xe_may_cu.png", width=80)
    st.sidebar.title("🚀 Menu")

    menu = st.sidebar.radio(
        "Click chọn nội dung:",
        [
            "Giới thiệu",
            "Báo cáo đánh giá",
            "Các phân khúc xe",
            "Tìm kiếm xe",
            "Thông tin nhóm thực hiện"
        ]
    )

    # ---------- BUSINESS PROBLEM ----------
    if menu == "Giới thiệu":
        st.title("📌 Giới thiệu tổng quan")
        st.markdown(
            """
            ### Bối cảnh
            Người mua xe máy cũ gặp nhiều khó khăn vì thị trường đa dạng, giá chênh lệch và thông tin thiếu minh bạch. nền tảng giao dịch xe uy tín, là cầu nối tin cậy giữa người mua và người bán trên toàn quốc. 
            Với lợi thế “Dễ tìm - Dễ mua”, Chợ Tốt Xe không ngừng hoàn thiện dịch vụ với thông tin minh bạch, quy trình đăng tin đơn giản và khả năng tìm xe nhanh chóng, đúng nhu cầu.
            Thị trường xe máy tại Việt Nam đang phát triển mạnh mẽ với sự đa dạng về mẫu mã, phân khúc và thương hiệu, đáp ứng nhu cầu di chuyển ngày càng cao của người tiêu dùng. Tùy thuộc vào sở thích và nhu cầu sử dụng, bạn có thể chọn mua xe theo các phân loại như xe số, xe tay ga, xe côn tay hay xe moto phân khối lớn. Người dùng cũng có thể lựa chọn theo dung tích xe như xe 50cc, xe từ 100 - 175cc,... để phù hợp nhu cầu di chuyển của mình.

            ### Mục tiêu dự án
            - Xây dựng hệ thống gợi ý xe máy cũ phù hợp nhu cầu.
            - Sử dụng model gợi ý dựa trên nội dung để đề xuất xe tương tự xe cần tìm và phân cụm nhằm xác định xe thuộc phân khúc nào.
            - Hiển thị phân tích dữ liệu, báo cáo hiệu suất mô hình.
            """
        )

    # ---------- EVALUATION (A + C) ----------
    elif menu == "Báo cáo đánh giá":

        st.title("Báo cáo mô hình gợi ý dựa trên nội dung và phân cụm")

        st.markdown("""
        ## 📝 **BÁO CÁO MÔ HÌNH GỢI Ý DỰA TRÊN NỘI DUNG **

        ### 🎯 **1. Mục tiêu hệ thống**
        Hệ thống được xây dựng nhằm gợi ý các xe máy tương tự dựa trên thông tin mô tả của từng xe. Việc gợi ý **dựa hoàn toàn trên nội dung** của các cột Thương hiệu, Dòng xe, Mô tả chi tiết của các xe đã đăng tải.

        ---

        ### ⚙️ **2. Quy trình xây dựng mô hình**

        #### **2.1. Tiền xử lý dữ liệu**
        - Làm sạch văn bản: viết thường, loại bỏ ký tự đặc biệt, stopwords.
        - Chuẩn hóa nội dung mô tả.
        - Vector hóa dữ liệu phục vụ tính toán.

        #### **2.2. Các phương pháp vector hóa đã thử nghiệm**
        1. **Gensim TF-IDF**
        - Sử dụng TF-IDF, tính tương tự bằng Gensim Similarity.
        - Kết quả khá nhưng tốc độ không tối ưu khi dữ liệu lớn.

        2. **Sklearn TF-IDF + Cosine Similarity**
        - Tính toán nhanh.
        - Dễ triển khai, dễ lưu và tải mô hình.
        - Độ chính xác gợi ý cao và ổn định.

        ---

        ### 📊 **3. Đánh giá mô hình**

        | Tiêu chí | Gensim | Cosine Similarity |
        |---------|--------|--------------------|
        | Tốc độ xử lý cho 5 đề xuất| Trung bình 30.6718 giây| **Rất nhanh** 0.0101 giây |
        | Độ ổn định | Khá | **Tốt** |
        | Độ chính xác qua đánh giá các nội dung gợi ý và qua giá trị similarity trung bình | Tốt | **Tốt nhất** |
        """)
        st.image("sosanh.png")
        st.markdown("""
        ---

        ### 🏆 **4. Lý do chọn Cosine làm mô hình chính**
        - Nhanh, phù hợp dữ liệu lớn.
        - Độ chính xác gợi ý ổn định.
        - Phù hợp cho dạng dữ liệu mô tả xe máy.

        ---

        ### 🚀 **5. Kết luận**
        Trang web sử dụng **TF-IDF + Cosine Similarity** làm mô hình chính vì tính hiệu quả, chính xác và tốc độ cao, đảm bảo trải nghiệm tốt cho người dùng.

        """)



        st.markdown("""
        ## 📝 **BÁO CÁO MÔ HÌNH PHÂN CỤM **

        ### 🎯 **1. Mục tiêu hệ thống**
        Hệ thống được xây dựng nhằm phân cụm xe máy thành các cụm tương đồng dựa trên Thương hiệu, Dòng xe, Số km đi được và Dung tích xe.

        ---

        ### ⚙️ **2. Quy trình xây dựng mô hình**

       """)
        st.image("Mohinhphancum.png")
        
        st.markdown("""
        ---

        ### 📊 **3. Đánh giá mô hình**

        Theo giá trị Silhouette tính được giữa các mô hình, mô hình trên sklearn cho kết quả tốt hơn trên pyspark và Agglomerative Clustering cho giá trị tốt nhất.
        """)
        st.image("DGmohinhphancum.png")
        st.markdown("""
        ---

        ### 🏆 **4. Lý do chọn Agglomerative làm mô hình chính**
        - Giá trị Silhouette cho ra tốt nhất
        - Các cụm được phân rõ ràng, không bị chồng lấn.

        ---

        ### 🚀 **5. Kết luận**
        Trang web sử dụng **Aggomerative** làm mô hình chính vì các cụm được phân rõ ràng.

        """)

    # ---------- EVALUATION (A + C) ----------
    elif menu == "Các phân khúc xe":
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
            **Cụm 1:** Xe giá rẻ và số km đã đi > 300km.  
            **Cụm 2:** Xe phân khối lớn và giá cao.  
            """
        )

        st.subheader("📈 Phân bố giá theo từng cụm")
        fig2, ax2 = plt.subplots(figsize=(3, 2))
        for cl in df_cluster["cluster"].unique():
            ax2.hist(df_cluster[df_cluster.cluster == cl]["Giá"], alpha=0.5, label=f"Cluster {cl}")
        ax2.legend()
        ax2.set_title("Phân bố Giá theo từng cụm")
        ax2.set_xlabel("Giá (triệu VNĐ)")
        st.pyplot(fig2)


        fig3, ax3 = plt.subplots(figsize=(3, 2))
        for cl in df_cluster["cluster"].unique():
            ax3.hist(df_cluster[df_cluster.cluster == cl]["Số Km đã đi"], alpha=0.5, label=f"Cluster {cl}")
        # đổi nhãn trục hoành sang triệu km
        xticks = ax3.get_xticks()
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([f"{x/1_000_000:.1f}" for x in xticks])
        ax3.legend()
        ax3.set_title("Phân bố số km đã đi theo từng cụm")
        ax3.set_xlabel("Số Km đã đi (triệu km)")
        st.pyplot(fig3)


        st.image("namdungtichcluster.png")

    # ---------- RECOMMENDATION (B) ----------
    elif menu == "Tìm kiếm xe":
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
                    if cluster_value ==0:
                        st.success(f"🚗 Xe này thuộc **cụm {cluster_value}**: đa số xe thuộc phân khúc này, bao gồm các dòng xe thông dụng, số km đã đi ở mức trung bình, thuộc xe có phân khối < 175cc")
                    if cluster_value ==1:
                        st.success(f"🚗 Xe này thuộc **cụm {cluster_value}**: Bạn đang chọn xe có phân khúc giá thấp, tuy nhiên các xe này đã sử dụng rất nhiều, có số km đi được rất cao ")
                    if cluster_value ==2:
                        st.success(f"🚗 Xe này thuộc **cụm {cluster_value}**: Bạn đang chọn phân khúc xe hiếm và cao cấp, các xe thuộc phân khúc này thường mới và có quãng đường đi ít")
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
        
            ### Ngày thực hiện
            💻 22/11/2025
            """
        )


if __name__ == "__main__":
    main()
