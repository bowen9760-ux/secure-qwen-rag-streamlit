# RAG 智能客服

这是一个基于 Streamlit、LangChain、阿里云百炼 OpenAI 兼容接口和 Chroma 的本地 RAG 示例。项目包含两个独立入口：

- `app_qa.py`：面向访客的问答页面；
- `app_file_uploader.py`：受管理员令牌保护的知识库管理页面。

  <img width="1618" height="1345" alt="屏幕截图 2026-09-11 001517" src="https://github.com/user-attachments/assets/ca88c7d6-d9ba-4b3c-87a5-1cc441416599" />
  <hr>
  <img width="1005" height="1237" alt="屏幕截图 2026-09-11 001445" src="https://github.com/user-attachments/assets/5dd3740c-ecaf-44ca-98c4-efef2e985426" />



## 这版解决了什么

- 每个 Streamlit 浏览器会话生成独立 UUID，不再让所有访客共用 `user_01`；
- 支持清空当前会话，并同步删除服务端会话历史；
- 检索使用相关性阈值，无可靠资料时拒绝凭空回答；
- 多轮问题会先结合历史改写，再用于检索；
- 检索资料按“不可信数据”处理，降低知识库提示注入风险；
- 聊天历史采用文件锁和原子替换，避免并发覆盖或半写文件；
- 入库使用稳定文档 ID，可替换同名文件并从 `data/` 完整重建；
- 上传端要求管理员令牌，限制文件类型、大小、编码和空内容；
- 不再打印完整提示词、用户对话或知识库正文。
- 直接依赖和 Windows/Python 3.13 验证环境均已锁定，并提供真实 Chroma 离线测试。

## 环境要求

已验证 Python 3.13.9；Python 3.11/3.12 可使用 `requirements.txt` 重新解析兼容依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

`requirements.lock` 是本次在 Windows/Python 3.13.9 上通过测试的完整锁文件。其他 Python 或操作系统优先使用直接依赖文件 `requirements.txt`，验证后再为目标环境生成新的锁文件。

复制环境变量示例并填写真实值：

```powershell
Copy-Item .env.example .env
```

`.env` 至少需要：

```dotenv
DASHSCOPE_API_KEY=你的-DashScope-API-Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_CHAT_MODEL=qwen3.7-plus
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_ADMIN_TOKEN=一个足够长且随机的管理员令牌
```

应用只把 `.env` 作为本地开发便利配置；已有的系统环境变量优先，不会被 `.env` 覆盖。不要提交 `.env`，也不要把管理员令牌发送给普通问答用户。

## 首次启动与重建

本版本使用新的 Chroma 集合 `rag_v2`。历史集合 `rag` 会保留在磁盘中，但新检索链不会读取它，也不会自动迁移其中可能过期或重复的内容。

因此，升级后必须重建一次：

1. 检查 `data/`，只保留当前有效、经过审核、UTF-8 编码的 `.txt` 文件；
2. 启动知识库管理页；
3. 输入 `RAG_ADMIN_TOKEN`；
4. 勾选重建确认框，点击“确认重建 rag_v2”；
5. 重建成功后再开放问答页。

也可以在已经配置 `.env` 的命令行中重建：

```powershell
.\.venv\Scripts\python.exe knowledge_base.py
```

旧 `rag` 集合的保留是刻意隔离，不代表其内容仍然可信。确认 `rag_v2` 工作正常并完成备份后，才应通过专门的维护流程删除旧集合；不要直接删除整个 `chroma_db/` 目录。

## 运行

### Windows 一键启动

完成首次安装和 `.env` 配置后，可以直接双击项目根目录中的文件：

- `启动问答.bat`：启动问答页并打开 `http://localhost:8501`；
- `启动全部服务.bat`：同时启动问答页和知识库管理页，并分别打开 `8501`、`8502`。

启动器会检查虚拟环境与 API Key，并避免在端口已有服务时重复启动。每个新服务都有独立的 PowerShell 窗口；关闭对应窗口即可停止该服务。

### 命令行启动

问答端：

```powershell
.\.venv\Scripts\streamlit.exe run app_qa.py --server.port 8501
```

知识库管理端建议仅监听受信任网络或本机：

```powershell
.\.venv\Scripts\streamlit.exe run app_file_uploader.py --server.port 8502 --server.maxUploadSize 2
```

打开问答页面后，每个新的 Streamlit 会话会生成 UUID。该 UUID 用于隔离历史，但不是用户身份认证；浏览器会话结束后再次进入，可能得到新的会话。需要长期账号体系时，应在 Streamlit 前增加正式登录、授权和审计层。

## 日常更新

上传单个 `.txt` 文件时，只有点击“确认上传并入库”才会执行写入，页面 rerun 不会自动重复上传。同名文件的新内容会替换该来源的旧分片。

当删除文件、批量修改资料或调整切分参数时，应使用 `data/` 重建功能，而不是逐个上传。重建期间不要并发执行上传。

模型名称、向量维度、服务端点和切分配置都会进入分片 ID 或审计元数据。代码会拒绝把不同嵌入空间混入同一个 collection；更换嵌入模型、维度或端点时，先把 `RAG_COLLECTION_NAME` 改成新的名称，再对 `data/` 完整重建和回归。聊天模型默认使用 `qwen3.7-plus`；生产环境如需避免别名随平台升级而漂移，应固定到经过回归验证的日期快照。

## 常见故障

如果网页提示 `AllocationQuota.FreeTierOnly` 或“免费额度已用完”，说明百炼已在免费额度耗尽后拒绝模型调用，不是 Chroma 或检索代码故障。请前往百炼免费额度页面检查当前聊天模型的余额。需要继续调用时，完成账号认证和充值后关闭“免费额度用完即停（仅使用免费额度）”；关闭后会产生按量费用。也可以把 `RAG_CHAT_MODEL` 改成另一个仍有独立免费额度且当前地域支持的模型，保存 `.env` 后重启问答应用。

## 验证与评估

不需要 DashScope API Key 的离线测试：

```powershell
$env:ANONYMIZED_TELEMETRY = "False"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

测试覆盖正确切分、同名替换、重复上传幂等、目录同步、路径穿越、真实 Chroma 分数检索、多轮改写、确定性拒答、引用标识及有界会话历史。

重建真实 `rag_v2` 后，可运行只消耗嵌入调用、不调用聊天模型的检索回归：

```powershell
.\.venv\Scripts\python.exe evaluate_retrieval.py
```

评估用例位于 `eval_dataset.json`。脚本返回非零退出码代表至少一个来源召回或越界拒答用例失败；扩大业务问题集后，再据此校准 `similarity_score_threshold`。

## 数据与安全边界

- `chroma_db/`、会话历史、`.env` 和旧摘要文件均属于运行时或敏感数据，已加入 `.gitignore`；
- `data/` 是重建来源，应接受版本控制和人工审核；
- 管理员令牌只保护应用入口，生产环境仍应配合 HTTPS、网络访问控制、反向代理认证和日志审计；
- 知识库可能包含错误资料。上线前应建立固定问题集，评估检索召回、引用正确性、拒答率和多轮一致性；
- 切换嵌入模型、向量维度或模型端点后必须使用新的 collection 名称重建，不能复用或混入旧向量。
- 默认聊天历史只存在当前进程内；只有设置 `RAG_HISTORY_PERSIST=true` 才会写入哈希文件名的 `user_chat_history_v2/`。持久化内容是未加密 JSON，默认保留 7 天并限制消息条数；敏感业务应使用受控数据库和正式的数据生命周期策略。
