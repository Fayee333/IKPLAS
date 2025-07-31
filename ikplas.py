# ikplas.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="肺炎克雷伯菌肝脓肿进展为侵袭综合征风险预测",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 特征名称映射 - 修改为模型训练时的特征名称
FEATURE_MAPPING = {
    'SOFA': 'SOFA评分',
    'D-Dimer': 'D-二聚体 (mg/L)',  # 使用大写连字符格式
    'PLT': '血小板计数 (×10^9/L)',   # 使用全大写格式
    'Hb': '血红蛋白 (g/L)'
}

# 模型期望的特征顺序
MODEL_FEATURE_ORDER = ['Hb', 'PLT', 'D-Dimer', 'SOFA']

# ----------- 模型加载函数 -----------
@st.cache_resource
def load_model():
    """健壮的模型加载函数"""
    try:
        # 尝试多种可能的模型位置
         possible_paths = [
            Path("models") / "my_model.pkl",        # GitHub推荐位置
            Path("my_model.pkl"),                   # 根目录位置
            Path("app") / "models" / "my_model.pkl",# 多层级项目
            Path("pneumonia_app") / "my_model.pkl", # Streamlit Cloud结构
            Path("resources") / "my_model.pkl"      # 资源文件夹
        ]
        
        # 尝试查找并加载模型
    for model_path in possible_paths:
            if model_path.exists():
                logger.info(f"找到模型文件: {model_path}")
                model = joblib.load(model_path)
                logger.info("模型加载成功")
                return model
        
        # 所有路径都找不到文件
        logger.error(f"未找到模型文件。检查位置: {[str(p) for p in possible_paths]}")
        st.error("❌ 模型文件未找到 - 请确认部署设置")
        
        # 添加模型上传作为后备方案
        st.subheader("模型上传备选方案")
        uploaded_file = st.file_uploader("上传ikplas_model.pkl文件", type=["pkl", "joblib"])
        if uploaded_file:
            try:
                with st.spinner("处理上传文件中..."):
                    save_path = Path("uploaded_model.pkl")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    model = joblib.load(save_path)
                    st.success("模型上传并加载成功！")
                    return model
            except Exception as e:
                st.error(f"上传文件处理失败: {str(e)}")
                st.stop()
        
        st.stop()
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.stop()

# ----------- 用户输入界面 -----------
def user_input_features():
    with st.sidebar:
        st.header("⚕️ 患者参数输入")
        
        # 使用单列布局
        with st.expander("临床指标", expanded=True):
            sofa = st.slider('SOFA评分', 0, 24, 5, step=1, 
                            help="序贯器官衰竭评估(SOFA)评分范围0-24分，评分越高表示器官功能障碍越严重")
            d_dimer = st.number_input('D-二聚体 (mg/L)', 0.0, 20.0, 1.5, step=0.1, 
                                     help="正常值通常<0.5mg/L，升高提示高凝状态和纤溶活性增强")
            plt = st.number_input('血小板计数 (×10^9/L)', 0, 1000, 200, step=10, 
                                 help="正常范围125-350×10^9/L，降低提示凝血功能障碍或消耗增加")
            hb = st.number_input('血红蛋白 (g/L)', 30, 200, 110, step=5, 
                                help="正常范围男性130-175g/L，女性115-150g/L")

    # 使用模型期望的特征名称和顺序创建DataFrame
    input_data = {
        'SOFA': sofa,
        'D-Dimer': d_dimer,
        'PLT': plt,
        'Hb': hb
    }
    
    return input_data

# ----------- SHAP解释可视化 -----------
def plot_shap_explanation(model, input_df):
    try:
        # 检查是否为树模型
        if hasattr(model, 'tree_') or any(hasattr(model, est) for est in ['tree_', 'estimators_']):
            explainer = shap.TreeExplainer(model)
        else:
            # 使用特征中位数作为背景
            explainer = shap.KernelExplainer(model.predict, np.median(input_df.values.reshape(1, -1), 
                                         link=shap.links.logit))
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_df.values)
        
        # 处理多分类/二分类
        if isinstance(shap_values, list) and len(shap_values) > 1:
            base_value = explainer.expected_value[1]
            shap_vals = shap_values[1]
        else:
            base_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]
            shap_vals = shap_values
        
        # 创建可视化
        plt.figure(figsize=(10, 4))
        shap.force_plot(
            base_value=base_value,
            shap_values=shap_vals,
            features=input_df.values[0],
            feature_names=input_df.columns.tolist(),
            matplotlib=True,
            show=False,
            text_rotation=15,
            plot_cmap=['#ff0051', '#008bfb']
        )
        
        plt.tight_layout()
        plt.gcf().set_facecolor('white')
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"SHAP解释生成失败: {str(e)}", exc_info=True)
        st.error(f"特征解释生成失败: {str(e)}")
        return None

# ----------- 主界面 -----------
def main():
    st.title("肺炎克雷伯菌肝脓肿进展为侵袭综合征风险预测")
    st.markdown("---")
    
    # 添加病情背景说明
    with st.expander("ℹ️ 疾病背景信息", expanded=False):
        st.write("""
        ​**侵袭性肺炎克雷伯菌肝脓肿综合征(IKPLAS)​**​ 是一种严重的感染性疾病，特征为：
        - 肝脓肿伴有远处转移性感染
        - 常见转移部位：眼内炎、脑膜炎、肺部感染等
        - 高危因素：SOFA评分升高、D-二聚体升高、血小板减少、贫血等
        
        本模型使用以下指标预测IKPLAS发生风险：
        1. SOFA评分 - 评估器官功能障碍程度
        2. D-二聚体 - 反映高凝状态和纤溶活性
        3. 血小板计数 - 评估凝血功能和炎症状态
        4. 血红蛋白 - 反映贫血程度和组织氧合能力
        """)
    
    # 加载模型
    model = load_model()
    
    # 获取输入
    input_dict = user_input_features()
    
    # 显示参数（使用漂亮的表格） - 使用UI友好的显示名称
    with st.expander("📋 当前输入参数", expanded=True):
        # 创建更友好的显示名称
        display_data = {
            "参数": [FEATURE_MAPPING.get(col, col) for col in input_dict.keys()],
            "数值": list(input_dict.values())
        }
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    
    # 预测按钮居中显示
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 开始风险评估", 
                               use_container_width=True, 
                               type="primary")
    
    # 预测结果展示
    if predict_btn:
        with st.spinner('正在分析参数...'):
            try:
                # 按照模型期望的顺序准备输入数据
                model_input = pd.DataFrame(
                    [[input_dict['Hb'], input_dict['PLT'], input_dict['D-Dimer'], input_dict['SOFA']]],
                    columns=MODEL_FEATURE_ORDER
                )
                
                st.info(f"模型输入特征顺序: {model_input.columns.tolist()}")
                
                # 预测概率
                proba = model.predict_proba(model_input)[0][1]
                
                # 根据概率划分为三个风险等级
                if proba < 0.3:
                    risk_level = "低风险"
                    color = "#2ECC71"  # 绿色
                    risk_percentage = f"{(proba*100):.1f}%"
                    severity = "低"
                elif proba < 0.7:
                    risk_level = "中风险"
                    color = "#F39C12"  # 橙色
                    risk_percentage = f"{(proba*100):.1f}%"
                    severity = "中"
                else:
                    risk_level = "高风险"
                    color = "#E74C3C"  # 红色
                    risk_percentage = f"{(proba*100):.1f}%"
                    severity = "高"
                
                # 显示结果卡片
                st.markdown("---")
                st.subheader("预测结果")
                
                # 风险卡片（居中）
                _, center, _ = st.columns([1, 3, 1])
                with center:
                    st.markdown(f"""
                    <div style="border-radius: 15px; padding: 25px; background-color: {color}10; 
                                border-left: 8px solid {color}; margin: 20px 0; text-align: center;">
                        <h3 style="color: {color}; margin-top:0;">{risk_level}</h3>
                        <div style="font-size: 3rem; font-weight: bold; color: {color}; margin: 10px 0;">
                            {risk_percentage}
                        </div>
                        <div style="font-size: 1rem; color: #555;">
                            侵袭综合征发生概率 (阈值: 30/70%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 特征重要性分析
                st.subheader("📈 特征贡献分析")
                fig = plot_shap_explanation(model, model_input)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    st.caption("""
                    ​**解释指南**:
                    - → 红色箭头表示增加风险的因素
                    - ← 蓝色箭头表示降低风险的因素
                    - 箭头长度代表影响大小
                    """)
                else:
                    st.warning("无法生成特征解释图")
                
                # 临床建议
                st.markdown("---")
                st.subheader("🩺 临床管理建议")
                
                if severity == "高":
                    st.error("""
                    ​**🔴 高风险管理方案**:
                    1. ​**立即收入ICU**进行高级生命支持
                    2. ​**强化抗感染治疗**​：碳青霉烯类+替加环素/多粘菌素
                    3. ​**影像学紧急评估**​：全身增强CT，重点排除眼内、中枢神经系统感染
                    4. ​**凝血功能管理**​：低分子肝素抗凝，监测DIC指标
                    5. ​**血液科会诊**​：评估是否需血小板输注
                    6. ​**多学科会诊**​：感染科、肝胆外科、重症医学科
                    """)
                elif severity == "中":
                    st.warning("""
                    ​**🟠 中风险管理方案**:
                    1. ​**住院治疗**并密切监测生命体征
                    2. ​**标准抗感染方案**​：三代头孢+氨基糖苷类
                    3. ​**每日评估**神经症状、视觉变化、呼吸道症状
                    4. ​**凝血功能监测**​：每日D-二聚体、FDP
                    5. ​**影像学复查**​：3天内肝脏CT/MRI评估脓肿变化
                    6. ​**营养支持**​：高蛋白、富含维生素饮食
                    """)
                else:
                    st.success("""
                    ​**🟢 低风险管理方案**:
                    1. ​**门诊随访**每周1-2次
                    2. ​**口服抗生素治疗**​：氟喹诺酮类+甲硝唑
                    3. ​**家庭监测**​：体温、腹部症状、视觉变化
                    4. ​**生活方式干预**​：控制血糖(如有糖尿病)，避免饮酒
                    5. ​**实验室复查**​：每周血常规、CRP、肝功能
                    6. ​**2周后影像学复查**​：肝脏超声
                    """)
                
                # 添加下载报告功能
                st.download_button(
                    label="📥 下载临床报告",
                    data=f"""
                    肺炎克雷伯菌肝脓肿侵袭综合征风险评估报告\n
                    患者风险等级: {risk_level} ({risk_percentage})\n
                    推荐管理方案: {severity}风险方案\n\n
                    输入参数:\n
                    {pd.DataFrame(display_data).to_string(index=False)}
                    """,
                    file_name=f"IKPLAS_风险评估_{risk_level}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                logger.error(f"预测失败: {str(e)}", exc_info=True)
                st.error("预测错误 - 请检查输入参数或模型")
                st.info("技术细节错误:")
                st.code(str(e))
                
                # 显示调试信息
                st.info(f"输入字典内容: {input_dict}")
                if 'model_input' in locals():
                    st.info("模型输入数据预览:")
                    st.dataframe(model_input)
                    
                    if hasattr(model, 'feature_names_in_'):
                        st.info(f"模型期望的特征名称: {model.feature_names_in_}")

# 主函数执行
if __name__ == '__main__':
    main()
