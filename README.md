<div align="center">

<img width="256" src="https://github.com/user-attachments/assets/6f9e4cf9-912d-4faa-9d37-54fb676f547e">
<br><br><br>

<h1> 🍌 蕉幻 (Banana Slides) </h1>

*Vibe your PPT like vibing code.*

<p>

[![GitHub Stars](https://img.shields.io/github/stars/Anionex/banana-slides?style=square)](https://github.com/Anionex/banana-slides/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Anionex/banana-slides?style=square)](https://github.com/Anionex/banana-slides/network)
[![GitHub Watchers](https://img.shields.io/github/watchers/Anionex/banana-slides?style=square)](https://github.com/Anionex/banana-slides/watchers)

[![Version](https://img.shields.io/badge/version-v0.1.0-4CAF50.svg)](https://github.com/Anionex/banana-slides)
![Docker](https://img.shields.io/badge/Docker-Build-2496ED?logo=docker&logoColor=white)
[![License](https://img.shields.io/github/license/Anionex/banana-slides?color=FFD54F)](https://github.com/Anionex/banana-slides/blob/main/LICENSE)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-42b883.svg)

</p>
  
一个基于nano banana pro的原生AI PPT生成应用，支持想法/大纲/页面描述生成完整PPT演示文稿、文本图片链接自动提取、上传任意素材、口头提出修改，迈向真正的“Vibe PPT”

*如果该项目对你有用, 欢迎star🌟 & fork🍴*
</p>

</div>


## ✨ 项目缘起
本人不擅长PPT制作，每次做PPT都需要花费大量时间在ppt样式调整和排版上。使用过传统的AI PPT app，虽然能快速产出ppt，但是还存在以下问题：
- 1️⃣只能选择预设模版，无法灵活调整风格
- 2️⃣自由度低，多轮改动难以进行 
- 3️⃣成品观感相似，同质化严重

有以上这些缺陷，难以满足我们“快”和“美”的两大PPT制作需求。即使自称Vibe PPT，但是在我的眼中还远不够“Vibe”。
但是nano banana🍌模型的出现让一切有了转机。我尝试使用🍌pro进行ppt页面生成，发现生成的结果，无论是质量，美感还是一致性，都做的非常好，且几乎能精确渲染所有prompt要求的文字+遵循参考图的风格： 好像真的能生成实用的结果了。那为什么不基于它的这一能力，做一个文生图的"Vibe PPT"应用呢？


## 🎯 功能介绍

### 亮点

- 🚀 **一句话生成 PPT**：从一个简单的想法快速得到大纲、页面描述和最终的 PPT 文稿
- 🔄 **三种生成路径**：支持从「想法 / 大纲 / 页面描述」三种方式起步，适配不同创作习惯
- 🔍 **文本与链接自动提取**：支持从一段文本中自动抽取要点、图片链接等信息
- 🧾 **上传任意素材**：可上传参考图片、示例 PPT 等作为风格和内容参考
- 🧙‍♀️ **AI 辅助编排**：由 LLM 生成结构清晰的大纲和逐页内容描述
- 🖼️ **高质量页面生成**：基于 nano banana pro🍌 生成高清、风格统一的页面设计
- 🗣️ **自然语言修改**：支持对单页或整套 PPT 进行「口头」式自然语言修改与重生成
- 📊 **一键导出**：自动组合为 PPTX / PDF，16:9 比例，开箱即用

### 1. 多种创建方式
- **从构想生成**：输入一句话 / 一段想法，自动生成完整大纲和页面内容
- **从大纲生成**：粘贴已有大纲，AI 帮你扩展为逐页详细描述
- **从描述生成**：直接提供每页描述，快速生成成品页面图片

### 2. 智能大纲与页面描述生成
- 根据用户输入主题自动生成 PPT 大纲与整套页面结构
- 以卡片形式呈现，支持删除、拖拽、调整顺序
- 既可以一次性批量生成，也可以单个编辑逐步补充和细化
- 内置并行处理能力，提升多页生成速度

### 4. 文本与素材理解
- 支持对输入文本进行关键点抽取、结构化整理
- 自动识别并提取其中的图片、链接等资源
- 支持上传参考图片、截图、旧 PPT 作为风格与内容线索

### 5. 多格式导出
- **PPTX 导出**：标准 PowerPoint 格式，可继续二次编辑
- **PDF 导出**：适合快速分享和展示
- 默认 16:9 比例，保证在主流显示设备上的观感


## 📦 使用方法

### 使用 Docker Compose🐳（推荐）
这是最简单的部署方式，可以一键启动前后端服务。

<details>
  <summary>📒Windows用户说明</summary>

如果你使用 Windows, 请先安装 Windows Docker Desktop，检查系统托盘中的 Docker 图标，确保 Docker 正在运行，然后使用相同的步骤操作。

> **提示**：如果遇到问题，确保在 Docker Desktop 设置中启用了 WSL 2 后端（推荐），并确保端口 3000 和 5000 未被占用。

</details>

0. **克隆代码仓库**
```bash
git clone https://github.com/Anionex/banana-slides
cd banana-slides
```

1. **配置环境变量**

创建 `.env` 文件（参考 `env.example`）：
```bash
cp env.example .env
```

编辑 `.env` 文件，配置必要的环境变量：
```env
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_API_BASE=https://generativelanguage.googleapis.com
...
```

2. **启动服务**

```bash
docker-compose up -d
```

3. **访问应用**

- 前端：http://localhost:3000
- 后端 API：http://localhost:5000


4. **查看日志**

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看后端日志
docker-compose logs -f backend

# 查看前端日志
docker-compose logs -f frontend
```

5. **停止服务**

```bash
docker-compose down
```

### 从源码部署

#### 环境要求
- Python 3.10 或更高版本
- [uv](https://github.com/astral-sh/uv) - Python 包管理器
- Node.js 16+ 和 npm
- 有效的 Google Gemini API 密钥

#### 后端安装

0. **克隆代码仓库**
```bash
git clone https://github.com/Anionex/banana-slides
cd banana-slides
```

1. **安装 uv（如果尚未安装）**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **安装依赖**

在项目根目录下运行：
```bash
uv sync
```

这将根据 `pyproject.toml` 自动安装所有依赖。

3. **配置环境变量**

复制环境变量模板：
```bash
cp .env.example .env
```

编辑 `.env` 文件，配置你的 API 密钥：
```env
GOOGLE_API_KEY=your-api-key-here
GOOGLE_API_BASE=https://generativelanguage.googleapis.com
PORT=5000
```

#### 前端安装

1. **进入前端目录**
```bash
cd frontend
```

2. **安装依赖**
```bash
npm install
```

3. **配置API地址**

前端会自动连接到 `http://localhost:5000` 的后端服务。如需修改，请编辑 `src/api/client.ts`。


#### 启动后端服务

```bash
cd backend
uv run python app.py
```

后端服务将在 `http://localhost:5000` 启动。

访问 `http://localhost:5000/health` 验证服务是否正常运行。

#### 启动前端开发服务器

```bash
cd frontend
npm run dev
```

前端开发服务器将在 `http://localhost:5173` 启动。

打开浏览器访问即可使用应用。


## 🛠️ 技术架构

### 前端技术栈
- **框架**：React 18 + TypeScript
- **构建工具**：Vite 5
- **状态管理**：Zustand
- **路由**：React Router v6
- **UI组件**：Tailwind CSS
- **拖拽功能**：@dnd-kit
- **图标**：Lucide React
- **HTTP客户端**：Axios

### 后端技术栈
- **语言**：Python 3.10+
- **框架**：Flask 3.0
- **包管理**：uv
- **数据库**：SQLite + Flask-SQLAlchemy
- **AI能力**：Google Gemini API
- **PPT处理**：python-pptx
- **图片处理**：Pillow
- **并发处理**：ThreadPoolExecutor
- **跨域支持**：Flask-CORS

## 📁 项目结构

```
banana-slides/
├── frontend/                    # React前端应用
│   ├── src/
│   │   ├── pages/              # 页面组件
│   │   │   ├── Home.tsx        # 首页（创建项目）
│   │   │   ├── OutlineEditor.tsx    # 大纲编辑页
│   │   │   ├── DescriptionEditor.tsx # 描述编辑页
│   │   │   └── Preview.tsx     # 预览和导出页
│   │   ├── components/         # UI组件
│   │   │   ├── outline/        # 大纲相关组件
│   │   │   ├── description/    # 描述相关组件
│   │   │   └── common/         # 通用组件
│   │   ├── store/              # Zustand状态管理
│   │   │   └── useProjectStore.ts
│   │   ├── api/                # API接口
│   │   │   ├── client.ts       # Axios客户端配置
│   │   │   └── endpoints.ts    # API端点
│   │   ├── types/              # TypeScript类型定义
│   │   └── utils/              # 工具函数
│   ├── public/                 # 静态资源
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                    # Flask后端应用
│   ├── app.py                  # Flask应用入口
│   ├── config.py               # 配置文件
│   ├── requirements.txt        # Python依赖
│   ├── models/                 # 数据库模型
│   │   ├── project.py          # Project模型
│   │   ├── page.py             # Page模型
│   │   └── task.py             # Task模型
│   ├── services/               # 服务层
│   │   ├── ai_service.py       # AI服务（基于demo.py）
│   │   ├── file_service.py     # 文件管理
│   │   ├── export_service.py   # PPTX/PDF导出
│   │   └── task_manager.py     # 异步任务管理
│   ├── controllers/            # API控制器
│   │   ├── project_controller.py
│   │   ├── page_controller.py
│   │   ├── template_controller.py
│   │   ├── export_controller.py
│   │   └── file_controller.py
│   ├── utils/                  # 工具函数
│   │   ├── response.py         # 统一响应格式
│   │   └── validators.py       # 数据验证
│   └── instance/               # SQLite数据库（自动生成）
│
├── uploads/                    # 文件上传目录
│   └── {project_id}/
│       ├── template/           # 模板图片
│       └── pages/              # 生成的PPT页面图片
│
├── demo.py                     # 原始demo（已集成到后端）
├── gemini_genai.py             # Gemini API封装（已集成）
├── pyproject.toml              # Python项目配置（使用 uv 管理依赖）
├── uv.lock                     # uv依赖锁定文件
├── docker-compose.yml          # Docker Compose 配置
├── .dockerignore               # Docker 忽略文件
├── LICENSE                     # MIT许可证
└── README.md                   # 本文件
```


## 🤝 贡献指南

欢迎通过
[Issue](https://github.com/Anionex/banana-slides/issues)
和
[Pull Request](https://github.com/Anionex/banana-slides/pulls)
为本项目贡献力量！

## 📄 许可证

MIT



