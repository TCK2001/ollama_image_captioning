import streamlit as st
import ollama
import tempfile
import PIL.Image

st.set_page_config(page_title="🐱 Vision Model Playground", layout="centered")

st.title("LLaVA: Image Captioning & Visual Q&A Practice")
st.write("使用 Ollama 模型與圖片互動 👇")

# ========== PART 0: setting ==========
with st.sidebar:
    st.header("⚙️ 模型設定")

    vision_model = st.text_input("🔍 基礎模型名稱", value="llava:7b")
    image_file = st.file_uploader("🖼️ 上傳圖片", type=["jpg", "png", "jpeg"])

    st.markdown("---")
    st.subheader("🗨️ Part 2 - Image Captioning")
    caption_prompt = st.text_input("輸入圖片說明請求語句", value="Describe this image:")

    st.subheader("❓ Part 3 - Visual Question Answering")
    vqa_prompt = st.text_input("輸入視覺問答問題", value="What color is the cat in the image?")

    st.markdown("---")
    st.subheader("🧑‍🎨 Part 4 - 自訂角色")
    custom_model_name = st.text_input("自訂模型名稱", value="dog-lover")
    system_prompt = st.text_area("System Prompt", value="You are a dog cuteness expert. Describe the dog’s appearance, mood, and give it a fun nickname.")
    custom_user_prompt = st.text_area("輸入對自訂模型的提問", value="Can you analyze this dog photo, explain what makes the dog cute, describe its personality, and give it a fun or adorable nickname? Please keep it simple.")
    custom_image_file = st.file_uploader("📸 上傳自訂模型使用圖片", type=["jpg", "png", "jpeg"], key="custom_image")
    
# ========== PART 1: 模型準備 ==========
st.markdown("## 🔄 模型下載與準備")
with st.spinner(f"正在拉取模型 {vision_model}..."):
    ollama.pull(vision_model)
    info = ollama.show(vision_model)
    st.success(f"✅ 模型 `{vision_model}` 準備完成！")

# ========== PART 2~4: 執行模型互動 ==========
if image_file:
    image = PIL.Image.open(image_file)
    st.image(image, caption="上傳圖片", use_column_width=True)

    # image save
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file, format="PNG")
        image_path = tmp_file.name

    # Part 2 - Image Captioning
    st.markdown("## 🖼️ Part 2 - Image Captioning")
    if st.button("🚀 執行 Captioning"):
        res2 = ollama.chat(
            model=vision_model,
            messages=[{
                'role': 'user',
                'content': caption_prompt,
                'images': [image_path]
            }]
        )
        st.session_state.caption_result = res2['message']['content']
        
    if 'caption_result' in st.session_state:
        st.success("✅ Captioning 完成")
        st.text_area("📄 Caption 結果", st.session_state.caption_result, height=100)

    # Part 3 - Visual Q&A
    st.markdown("## ❓ Part 3 - Visual Question Answering")
    if st.button("🚀 執行 VQA"):
        res3 = ollama.chat(
            model=vision_model,
            messages=[{
                'role': 'user',
                'content': vqa_prompt,
                'images': [image_path]
            }]
        )
        st.session_state.vqa_result = res3['message']['content']
        
    if 'vqa_result' in st.session_state:
        st.success("✅ VQA 完成")
        st.text_area("📄 VQA 結果", st.session_state.vqa_result, height=100)

    # Part 4 - Custom Role
    st.markdown("## 🧑‍🎨 Part 4 - Custom Role Interaction")
    if custom_image_file:
        custom_image = PIL.Image.open(custom_image_file)
        st.image(custom_image, caption="自訂模型使用圖片", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            custom_image.save(tmp_file, format="PNG")
            custom_image_path = tmp_file.name

        if st.button("🚀 執行 Custom 模型"):
            with st.spinner(f"正在建立模型 {custom_model_name}..."):
                ollama.create(model=custom_model_name, from_=vision_model, system=system_prompt)

            res4 = ollama.chat(
                model=custom_model_name,
                messages=[{
                    'role': 'user',
                    'content': custom_user_prompt,
                    'images': [custom_image_path]
                }]
            )
            st.session_state.custom_result = res4['message']['content']

        if 'custom_result' in st.session_state:
            st.success("✅ 自訂模型回應完成")
            st.text_area("📄 Custom 模型結果", st.session_state.custom_result, height=150)
    else:
        st.info("📎 請上傳用於自訂模型的圖片")
        
    st.markdown(
    '<div style="text-align: center; color: gray; font-size: 0.9em;">'
    'Made by <a href="https://medium.com/@hichengkang" target="_blank" style="color: gray; text-decoration: none;">tck</a>'
    '</div>',
    unsafe_allow_html=True
)
else:
    st.warning("請先上傳一張圖片才能執行模型。")
    st.markdown(
    '<div style="text-align: center; color: gray; font-size: 0.9em;">'
    'Made by <a href="https://medium.com/@hichengkang" target="_blank" style="color: gray; text-decoration: none;">tck</a>'
    '</div>',
    unsafe_allow_html=True
    )
