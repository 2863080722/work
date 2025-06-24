# Key News - 个性化新闻推荐系统

本项目是一个基于深度学习的个性化新闻推荐系统。它利用多模态内容理解和动态用户画像技术，为用户提供精准、个性化的新闻内容。系统特别关注"一带一路"倡议相关国家的新闻。

## 目录

- [项目核心功能](#项目核心功能)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [上手指南](#上手指南)
  - [环境配置](#环境配置)
  - [运行](#运行)
- [注意事项](#注意事项)

## 项目核心功能

系统的核心是位于 `keynews.py` 中的推荐引擎，主要包括以下几个模块：

1.  **多模态内容编码器 (`MultiModalContentEncoder`)**:
    *   使用 `BERT` 处理新闻文本。
    *   使用 `CLIP` 处理新闻图片。
    *   结合一个3D卷积网络处理视频。
    *   通过跨模态注意力机制融合文本、图像和视频特征，生成统一的内容向量。

2.  **用户画像分析器 (`UserProfileAnalyzer`)**:
    *   动态分析用户的互动数据（如点击、阅读时长等）。
    *   构建包含用户兴趣、阅读习惯和国家/地区偏好的复杂用户画像。

3.  **增强用户模型 (`EnhancedUserModel`)**:
    *   使用 `LSTM` 捕捉用户的短期兴趣。
    *   使用 `Transformer` 建模用户的长期兴趣。
    *   融合用户画像特征，生成动态的用户向量。

4.  **个性化新闻推荐器 (`PersonalizedNewsRecommender`)**:
    *   整合内容向量和用户向量。
    *   利用图神经网络（GNN）进行知识图谱增强。
    *   最终通过一个预测层计算新闻的推荐分数，并进行排序。

## 技术栈

*   **后端 / 核心模型**: Python, PyTorch, Transformers, scikit-learn
*   **数据处理**: pandas, numpy
*   **媒体处理**: OpenCV, Pillow
*   **前端 (静态页面)**: HTML, CSS, JavaScript, jQuery

## 项目结构

```
.
├── keynews.py                # 核心推荐系统逻辑
├── requirements.txt          # Python 依赖
├── login and register/       # 登录和注册页面 (静态)
│   ├── login.html
│   └── register.html
├── 首页/                     # 主页界面 (静态)
│   └── index.html
└── system/                   # 项目组件的容器目录
    ├── html轮播/             # 一个HTML轮播组件
    └── ... (包含其他前端页面的副本)
```

**注意**: 当前项目结构存在一些冗余。例如，`login and register` 和 `首页` 目录在根目录和 `system/` 目录下重复出现。核心的Python逻辑位于根目录的 `keynews.py` 中，而 `system/main.py` 是一个未使用的模板文件。

## 上手指南

### 环境配置

1.  克隆或下载本项目。
2.  建议创建一个Python虚拟环境。
3.  安装所需的依赖包：
    ```sh
    pip install -r requirements.txt
    ```

### 运行

当前项目似乎缺少一个Web框架（如 Flask 或 Django）来连接后端逻辑和前端页面。要完整运行此项目，您需要：

1.  **实现一个Web服务**:
    *   选择一个Web框架（例如 Flask）。
    *   创建一个 `app.py` 文件。
    *   在 `app.py` 中，导入并实例化 `NewsRecommendationSystem`。
    *   编写API端点，例如：
        *   `/login` 和 `/register` 用于处理用户认证。
        *   `/recommend` 用于接收用户ID并调用推荐系统，返回新闻列表。
        *   `/` 用于提供主页和其他静态页面。

2.  **整合前端与后端**:
    *   修改前端的JavaScript代码，使其能够调用您在步骤1中创建的API。
    *   将从后端获取的新闻数据动态地渲染到前端页面上。

## 注意事项

*   `keynews.py` 中的模型（如BERT, CLIP）需要从Hugging Face Hub下载预训练权重。首次运行时请确保网络连接正常。
*   项目中的前端部分是静态的，需要进一步开发以实现与后端逻辑的完整交互。
*   请根据您的实际需求，整合和清理项目中重复的前端目录。




