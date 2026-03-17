# 支持 OpenAI 兼容模型 API 实施计划

## 1. 目标
在 `gemini-cli` 项目中增加对 OpenAI 兼容接口（如 OpenAI, DeepSeek, 各种本地大模型等）的支持。
允许用户配置：
- API Base URL (调用地址)
- Model Name (模型名称)
- API Key (密钥)
- Temperature (温度) 等参数
**核心约束**：保持整个项目现有的工具调用（Tool Calling）逻辑和架构完全不改变。

## 2. 架构设计：适配器模式 (Adapter Pattern)
项目目前通过 `ContentGenerator` 接口抽象了底层的大模型调用。现有的工具调用逻辑构建的是 `@google/genai` 的 `GenerateContentParameters` 对象。
为了不改变上层逻辑，我们将创建一个 `OpenAIContentGenerator` 类，实现 `ContentGenerator` 接口。该类将在内部负责：
1. 将 Gemini 格式的请求（包含 Tools, System Instructions, Contents）转换为 OpenAI Chat Completion 格式。
2. 将 OpenAI 格式的响应（包含 Tool Calls, Content）转换回 Gemini 的 `GenerateContentResponse` 格式。

## 3. 具体实施步骤

### 步骤 1: 更新配置定义 (Settings Schema & Types)
- **文件**: `/Users/barry/Downloads/code/gemini-cli/packages/core/src/core/contentGenerator.ts` (或定义 AuthType 的文件)
  - 在 `AuthType` 枚举中新增 `OPENAI_COMPATIBLE = 'openai-compatible'`。
- **文件**: `/Users/barry/Downloads/code/gemini-cli/packages/cli/src/config/settingsSchema.ts`
  - 在配置 Schema 中新增针对 OpenAI 兼容接口的配置项：
    - `customApi.baseUrl` (string)
    - `customApi.modelName` (string)
    - `customApi.apiKey` (string)
    - `customApi.temperature` (number)

### 步骤 2: 实现 OpenAI 适配器 (OpenAIContentGenerator)
- **新建文件**: `/Users/barry/Downloads/code/gemini-cli/packages/core/src/core/openaiContentGenerator.ts` (或类似路径)
- **实现内容**:
  - 创建 `OpenAIContentGenerator` 类，实现 `ContentGenerator` 接口。
  - 引入 `openai` 官方 SDK 或使用 `fetch` 直接调用标准接口。
  - **请求转换逻辑**: 编写转换函数，将 `GenerateContentParameters` 映射为 OpenAI 的 `messages` 数组，并将 Gemini 的 `tools` 定义转换为 OpenAI 的 `tools` (JSON Schema) 定义。
  - **响应转换逻辑**: 编写转换函数，将 OpenAI 的返回结果（特别是 `tool_calls`）包装成符合 `@google/genai` 结构的 `GenerateContentResponse`。
  - 实现 `generateContent` 和 `generateContentStream` 方法。

### 步骤 3: 更新生成器工厂 (Generator Factory)
- **文件**: `/Users/barry/Downloads/code/gemini-cli/packages/core/src/core/contentGenerator.ts`
  - 修改 `createContentGenerator` 工厂函数。
  - 当检测到配置的认证类型为 `OPENAI_COMPATIBLE` 时，读取 `customApi` 相关的配置（URL, Model, Key, Temperature），并实例化返回 `OpenAIContentGenerator`。

### 步骤 4: 更新 CLI 交互界面 (UI)
- **文件**: `/Users/barry/Downloads/code/gemini-cli/packages/cli/src/ui/auth/AuthDialog.tsx` (及相关 CLI 提示文件)
  - 在初始化或重新配置的认证选择界面中，添加“使用 OpenAI 兼容接口”的选项。
  - 当用户选择该选项时，引导用户输入 Base URL, Model Name, API Key 等必要信息，并保存到本地配置中。

## 4. 风险与注意事项
- **Tool Calling 格式差异**: Gemini 和 OpenAI 在 Tool Calling 的 JSON Schema 定义上可能存在细微差异，转换逻辑需要严谨，确保现有的工具定义能被 OpenAI 模型正确识别。
- **流式输出 (Streaming)**: `generateContentStream` 的转换相对复杂，需要处理 Server-Sent Events (SSE) 并将其组装成符合 Gemini 格式的 AsyncGenerator。

请确认此计划。确认后，我将开始按照步骤进行代码修改。