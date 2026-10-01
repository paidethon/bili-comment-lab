# bili-comment-lab — B 站评论提取 / AI 提炼工具链

无 GUI、可长期重复使用的评论抓取与 AI 分析管线，输出可直接交给 AI Agent
阅读的评论包，并用便宜的 flash/mini 模型做 map-reduce 提炼。

**核心原则：提炼，但不改变原意。** 原始数据（`data/raw/`）永远只读；
分析结果全部可追溯 `comment_id`。

## 快速开始

```bash
cd bili-comment-lab
cp .env.example .env        # 按需填 BILIBILI_COOKIE / LLM_*

./bili all "https://www.bilibili.com/video/BVxxxx"    # 全流程
# 或分步：
./bili crawl   "URL"      # 抓取 -> data/raw/<BVID>/comments.jsonl
./bili prepare "BVID"     # 清洗 + Agent Pack -> data/normalized/ + data/agent/
./bili analyze "BVID"     # map-reduce AI 分析 -> data/reports/
```

`prepare`/`analyze` 接受 BV 号或 `data/raw/` 下的目录名，可对已有数据反复重跑。


## 快速开始（Docker）

```bash
git clone https://github.com/<your-name>/bili-comment-lab.git
cd bili-comment-lab
cp .env.example .env                 # LLM_BASE_URL / LLM_API_KEY / LLM_MODEL；抓楼中楼填 BILIBILI_COOKIE
docker compose up -d --build
curl http://127.0.0.1:8765/api/health
# 打开 http://127.0.0.1:8765/ ；公网部署请套反代（TLS + Basic Auth）
```

容器约束：单 worker（任务串行是设计而非缺陷）、`mem_limit=400m`、`cpus=0.8`、
数据全部落在 `./data` 卷。容器以非 root（uid 10001）运行，首次部署先执行一次：

```bash
sudo chown -R 10001:10001 ./data
```

不用 Docker 时见下方「快速开始」的本机运行方式。

## 界面

单页应用（无构建步骤），内置浅色/深色双主题（右上角切换，跟随系统偏好，
localStorage 持久化）。设计令牌体系参考 shadcn/ui、Radix Colors、
Ant Design、Material 3、Tremor 等主流设计系统的实践。

## 底层选型（为什么是 BilibiliCrawler）

| 项目 | BilibiliCrawler（选用） | MediaCrawler |
| ---- | ---- | ---- |
| Headless | ✅ 纯 requests，零浏览器 | ❌ 强依赖 Playwright（buvid/cookie 注入走浏览器） |
| wbi 签名 | ✅ 完整实现 + nav key 缓存 | ✅ |
| 楼中楼 | ✅ /reply/reply 翻页 | ✅ |
| 置顶评论 | ❌ 上游缺口 → **本项目已补** | ✅（top/top_replies 提取） |
| Cookie | ✅ CLI 参数/环境变量注入 | 需 Playwright |
| 服务器内存 | ~50MB | headless Chrome 150-300MB+ |
| License | MIT | 非商用学习协议 |
| 维护 | 2026-09-16 提交 | 2026-09-19 提交 |

结论：以 **BilibiliCrawler（MIT，vendored 于 `vendor/`）** 为 API 层，
移植 MediaCrawler 的置顶评论提取思路补齐其唯一功能缺口
（`bilicom/crawl.py: fetch_pinned_comments`）。

## 数据流与目录

```
data/
├── raw/<BVID>/comments.jsonl     原始抓取（永不被 AI 修改）+ meta.json
├── normalized/<BVID>/            统一字段 schema + stats.json
├── agent/<BVID>/                 Agent Pack（manifest/video.md/part-*.md/statistics.json）
├── reports/<BVID>/               summary/topics/opinions/questions/suggestions/representative
└── cache/                        LLM 结果缓存（content hash + model + prompt version）
```

Normalized schema（隐私字段只保留盐化指纹）：

```json
{
  "video_id": "", "comment_id": 0, "parent_comment_id": null,
  "content": "", "created_at": "", "like_count": 0, "reply_count": 0,
  "is_pinned": false, "level": 1, "author_hash": ""
}
```

Agent Pack 的每个 part（默认 24k token 预算）都包含完整原文 blockquote，
不润色、不截断；单条超长评论独占一个 part，绝不丢弃。

## AI 分析（Layer 1 + Layer 2）

- **Layer 1（抽取式）**：每个分片独立抽取主题/支持/反对/问题/建议/情绪，
  全部携带 `comment_ids`，跨分片确定性合并（无 LLM 参与，零漂移）。
- **Layer 2（总结式）**：只在 Layer 1 之上生成报告，prompt 强制要求观点
  可追溯、不做事实裁决、保留少数派意见。
- **Map-Reduce**：分片 map → 主题聚合 → reduce。每步缓存于
  `data/cache/<model>/`，key = 内容 hash + model + prompt version，
  失败重跑只补缺失分片，不重复花钱。
- **Prompt 版本化**：`prompts/extract-v1.md` 等，结果记录
  model/prompt_version/timestamp/input_hash。
- 未配置 LLM 时自动降级为 extractive-only（统计 + 高赞代表评论），
  流程不中断。


## Web 工作台（bilicom-web）

`bilicom-web.service`（FastAPI，127.0.0.1:8765，公网经 Caddy + Basic Auth）
提供五个页面：**提取 / 分析 / 数据集 / 历史任务 / 设置**。

- **提取**：单视频；批量（多行 BV/URL，一行一个）；UP 主批量
  （`space.bilibili.com/<mid>` 或纯数字 mid → 最近 10 个视频）。
  支持运行中任务取消（阶段间生效）。
- **分析（无损精炼）**：核心是"精炼，但不改变评论原话"——

  ```
  原始评论（raw 只读）
      → 完全重复删除（规范化文本 sha）
      → 近重复折叠（字符 3-gram MinHash LSH + Jaccard 校验）
      → 语义向量（OpenAI 兼容 /embeddings，内容寻址缓存，断点续跑；
                  无 API 时回退本地 bigram TF-IDF）
      → 贪心 leader 聚类（余弦阈值，高赞优先成为簇心）
      → 代表评论（每簇 top-N 原文，AI 只能挑选 id，不能改写）
      → 可选 AI 标注（模式①：AI 仅输出 topic/keywords/representative_ids，
         id 不在簇内即丢弃；AI 生成的任何文字永远不会成为评论）
  ```

  全流程漏斗（原始→去重→折叠→聚类→代表→压缩率）、每簇关键词、
  预览（随机 100 条、零 API 成本）、取消/续跑/重新精炼、AI 用量与费用统计。
- **数据集**：所有已抓取/导入数据的统一浏览（SQLite 索引 data/lab.db，
  原始 JSONL 永不被写入）：层级/点赞/回复/关键词/日期筛选 + 分页；
  导入 JSONL/JSON/CSV（列名自动映射）；导出 raw jsonl/csv/json、
  精炼 jsonl/csv/md、以及「全部数据 ZIP」：
  `metadata.json + raw/{jsonl,csv,json} + refined/{jsonl,csv,md} + analysis/{topics,keywords,clusters,ai_report} + manifest.json`，
  隐私开关可隐藏用户名/UID/属地。
- **历史任务**：每次精炼的算法/模型/参数/统计/成本完整保留，可续跑、重跑。
- **设置 → AI / NLP**：OpenAI Compatible（OpenAI/DeepSeek/GLM/Kimi/Ollama…）。
  API Key 只存服务器（`data/webui/ai_settings.json`，0600），浏览器仅见掩码；
  支持测试连接、/v1/models 自动拉模型列表、任务级模型覆盖、价格→费用估算。
  本地 Ollama（`http://localhost:11434/v1`）免 Key。
- **三种模式的边界**：① 无损精炼（默认，AI 不生成评论文本）→
  精炼结果每行带 comment_id/root_id，展示文本永远从数据库原文读取；
  ② AI 综合报告（原 map-reduce，报告属于"AI 生成分析"层）；
  两者在界面上、导出目录中完全分开，绝不混淆。

AI 摘要（模式③/原 map-reduce）行为不变：`data/reports/`，
未配置 LLM 时自动降级 extractive-only。

## Cookie 与风控

- 匿名可抓，但国内机房 IP 实测**楼中楼接口会持续 412**（见
  `docs/smoke-test-412.md`）。对策只有一种：填登录 Cookie。
- 遇到持续 412/验证码：**停止抓取**。本工具不做也不会做风控规避。
- 合理限速由上游实现（自适应 0.1–2s 间隔、失败指数退避、最多 3 次重试）。

## 测试

```bash
./venv/bin/python -m pytest tests/ -q     # 63 个用例：schema/去重/父子关系/置顶/
                                          # 分块预算/缓存键/JSON 宽松解析/Markdown/emoji/
                                          # 数据集索引/无损精炼/近重复折叠/聚类/Provider 设置/
                                          # 文本清洗/表情桶/情感对账/立场消解/快照 ingest/隐私
```

集成测试用 fixture 数据离线跑完 normalize→prepare→analyze 全链路。

## 已知限制

- 楼中楼服务端强制每页 20 条（上游注释，ps 参数无效）——已按上游逻辑处理。
- 置顶评论的**楼中楼**依赖登录 Cookie 才能稳定抓取（412 问题同上）。
- `--limit`/`--max-pages` 只用于小规模测试；全量爬取请尊重平台与作者。

## 修复记录（2026-09-30 质量审计）

针对导出包审计发现的全部问题，管线层修复如下（`tests/test_fixes.py` 覆盖）：

| 问题 | 修复 |
|---|---|
| 回复前缀 `回复 @昵称：` 污染聚类与标签 | `refine.clean_for_analysis`：分析视图剥离前缀与 `[表情]`，原文不动 |
| 表情码主导话题（思考/微笑霸榜） | 表情码在向量化/关键词层归零；纯表情评论进独立「纯表情/打卡」桶 |
| 英文字母 n-gram 碎片簇（ar/oge） | `content_terms`：拉丁按整词（≥2 字符）入向量，不再切字母 bigram |
| 聚类碎裂（1192 簇，最多 35 条） | `choose_threshold` 触发门槛 3000→800，目标独立率 ≤35%，阈值可降到 0.05 |
| 标签截断（"会资源"）| 关键词加入 3/4-gram + 双向子串抑制；短拉丁标签兜底「杂项」 |
| quality_score 只是 log10(赞)，0 赞=0 分 | 0–1 综合分：55% 内容信号 + 45% 对数化点赞，每行必有值 |
| refined 丢失楼中楼结构（root_id 全空） | `run_cluster_members` SELECT 补 root_id/parent_id，导出字段含 parent_id |
| raw 与 refined 快照不一致（3403 vs 2671） | `ingest_video` 快照语义：删除不在当前 raw 文件中的残留行；bundle 校验并写 warnings |
| 立场标注自相矛盾（3 条同时支持+反对） | `textstats.resolve_stances`：按文本重合度归入单一立场，平票归中立 |
| 夸赞/倾诉混入论题立场清单 | 论题外反馈分流到「其他反馈」，不计入支持/反对声量 |
| 报告 5 个重复头部、代表性评论无原文 | analyze 写单头部 `report.md`；代表性评论查数据集补原文引用 |
| 情绪计数 1386+724+558≠2671 且不可复核 | `textstats.sentiment_counts` 词典法覆盖全部评论，三类之和恒等于总数 |
| 模型引用幻觉 ID（如 299746416209） | 引用 ID 全量对数据集校验，剔除并记录进 `qa.json`；报告尾部附「质检与口径」 |
| 隐私只清字段、正文 @昵称泄漏（690 条） | `_privacy_scrub` 连正文 `回复 @匿名：` 一起脱敏 |
| "可在界面中展开"指向不存在的界面 | 折叠说明改为指向 refined/comments.jsonl（按 cluster_id 筛选） |
| embeddings 缓存键与文本不一致 | 缓存键改为清洗后文本的 sha，输入与键同源 |

AI 报告 reduce prompt 升级为 `reduce-v2`（其他反馈小节、引用给定情绪计数、禁止同 ID 双立场）。
新导出样例：`data/webui/exports/BV1AQdoBgEa1_export_fixed_*.zip`；重建脚本 `regen_video.py`。
