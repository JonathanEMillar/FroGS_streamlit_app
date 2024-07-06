import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import io
import csv

def load_embeddings(file):
    content = file.getvalue().decode('utf-8')
    reader = csv.reader(io.StringIO(content))
    return {rows[0]: np.array(rows[1:], dtype=np.float32) for rows in reader}

def gene_sim_mean_std(go_emb, archs4_emb):
    all_genes = set(list(go_emb.keys()) + list(archs4_emb.keys()))

    def sim_mean_std(emb):
        emb_mat = []
        gene_ids = []
        for i, gid in enumerate(all_genes):
            if gid in emb:
                emb_mat.append(emb[gid])
                gene_ids.append(gid)

        emb_mat = np.vstack(emb_mat)
        sim_pairwise = cosine_similarity(emb_mat, emb_mat)
        sim_mean = np.mean(sim_pairwise, axis=1)
        sim_std = np.std(sim_pairwise, axis=1)

        mean_std_dict = {}
        for i, gid in enumerate(gene_ids):
            mean_std_dict[gid] = (sim_mean[i], sim_std[i])
        return mean_std_dict

    return sim_mean_std(go_emb), sim_mean_std(archs4_emb)

def compute_list_emb(glist, go_emb, archs4_emb, mean_std_dict_go, mean_std_dict_archs4):
    def compute_gene_weight(mat, sim_mean, sim_std):
        sim_pairwise = cosine_similarity(mat, mat)
        sim_pairwise[np.isnan(sim_pairwise)] = 0
        sim_sum = np.mean(sim_pairwise, axis=1)
        gene_weight = (sim_sum - sim_mean) / sim_std
        return gene_weight

    go_mat = np.zeros([len(glist), len(list(go_emb.values())[0])])
    for i, gid in enumerate(glist):
        if gid in go_emb:
            go_mat[i, :] = go_emb[gid]
        else:
            go_mat[i, :] = np.zeros(len(list(go_emb.values())[0]))

    archs4_mat = np.zeros([len(glist), len(list(archs4_emb.values())[0])])
    for i, gid in enumerate(glist):
        if gid in archs4_emb:
            archs4_mat[i, :] = archs4_emb[gid]
        else:
            archs4_mat[i, :] = np.zeros(len(list(archs4_emb.values())[0]))

    go_mat = normalize(go_mat)
    archs4_mat = normalize(archs4_mat)

    sim_mean_go = np.array([mean_std_dict_go[gid][0] if gid in mean_std_dict_go else 0 for gid in glist])
    sim_std_go = np.array([mean_std_dict_go[gid][1] if (gid in mean_std_dict_go and mean_std_dict_go[gid][1] > 0) else float('inf') for gid in glist])
    sim_mean_archs4 = np.array([mean_std_dict_archs4[gid][0] if gid in mean_std_dict_archs4 else 0 for gid in glist])
    sim_std_archs4 = np.array([mean_std_dict_archs4[gid][1] if (gid in mean_std_dict_archs4 and mean_std_dict_archs4[gid][1] > 0) else float('inf') for gid in glist])

    gene_weight_go = compute_gene_weight(go_mat, sim_mean_go, sim_std_go)
    gene_weight_a4chs4 = compute_gene_weight(archs4_mat, sim_mean_archs4, sim_std_archs4)
    gene_weight = np.maximum(gene_weight_go, gene_weight_a4chs4)
    gene_weight = np.clip(gene_weight, 0, 1)
    gene_weight = gene_weight.reshape((-1, 1))

    concatenated_mat = np.hstack((go_mat, archs4_mat))

    return np.sum(concatenated_mat * gene_weight, axis=0) / np.clip(np.sum(gene_weight), 1e-100, None)

def main():
    st.set_page_config(page_title="FroGS Embeddings", page_icon="🐸", layout="wide")

    st.title("FroGS Embeddings Generator")
    st.markdown("""
    Welcome to the FroGS (Functional Representation of Gene Signature) Embeddings Generator. 
    This tool helps you create embeddings for gene signatures using GO and ARCHS4 data.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.header("📁 File Upload")
        go_file = st.file_uploader("Upload GO embedding file (CSV)", type="csv", help="CSV file containing GO embeddings")
        archs4_file = st.file_uploader("Upload ARCHS4 embedding file (CSV)", type="csv", help="CSV file containing ARCHS4 embeddings")
        signature_file = st.file_uploader("Upload gene signatures file (TXT)", type="txt", help="TXT file containing gene signatures")

    with col2:
        if go_file:
            st.success(f"GO file uploaded: {go_file.name}")
        if archs4_file:
            st.success(f"ARCHS4 file uploaded: {archs4_file.name}")
        if signature_file:
            st.success(f"Signature file uploaded: {signature_file.name}")

    if st.button("Generate Embeddings", key="generate_button", help="Click to process files and generate embeddings"):
        if go_file and archs4_file and signature_file:
            try:
                with st.spinner("🔄 Loading embeddings..."):
                    go_emb = load_embeddings(go_file)
                    archs4_emb = load_embeddings(archs4_file)
                    st.success("✅ Embeddings loaded successfully.")

                with st.spinner("🔄 Computing mean and std..."):
                    mean_std_dict_go, mean_std_dict_archs4 = gene_sim_mean_std(go_emb, archs4_emb)
                    st.success("✅ Mean and std computed successfully.")

                with st.spinner("🔄 Processing signatures..."):
                    content = signature_file.getvalue().decode('utf-8')
                    lines = content.strip().split('\n')
                    data = []
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split(',')
                            signature_name = parts[0]
                            gene_list = parts[1:]
                            
                            embedding = compute_list_emb(gene_list, go_emb, archs4_emb, mean_std_dict_go, mean_std_dict_archs4)
                            
                            data.append([signature_name] + embedding.tolist())

                    df = pd.DataFrame(data, columns=['Signature'] + [f'Dim_{i}' for i in range(len(data[0])-1)])
                    
                    csv = df.to_csv(index=False)
                    
                    st.success(f"✅ Processing complete. Generated embeddings for {len(data)} signatures.")
                    
                    st.download_button(
                        label="📥 Download Embeddings CSV",
                        data=csv,
                        file_name="embeddings.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please upload all required files.")

    st.sidebar.header("About FroGS")
    st.sidebar.info("""
    FroGS is described in the original publication by **Chen, H., King, F.J., Zhou, B. et al.
Drug target prediction through deep learning functional representation of gene signatures.
Nature Commun. 15, 1853 (2024)**. <https://doi.org/10.1038/s41467-024-46089-y>.
    """)
    st.sidebar.header("App code")
    st.sidebar.info("[Github repo](https://github.com/JonathanEMillar/FroGS_streamlit_app)")

if __name__ == "__main__":
    main()