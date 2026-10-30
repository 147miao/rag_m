# DESIGN.md — RAG-M 检索控制台

> 遵循 OpenDesign 的 DESIGN.md 协议（品牌系统可移植）。本文件是唯一视觉事实来源：颜色、字体、间距、组件、状态、动效都在这里定义，代码只引用 token，不临时发明样式。

## 0. 产品定位

RAG-M 是一个**研究仪器**，不是聊天玩具。用户来这里是为了：问一个问题，看清系统走了哪条链路、用了哪些证据、代价是多少。
因此界面要传达三件事：**证据优先、链路可见、数字可信**。

- 气质：精确、克制、有编辑感（editorial × instrument）。
- 反 AI 味：不用紫色渐变、不用玻璃拟态、不用大圆角、不用 emoji 图标、不用"✨ AI 驱动"式文案。
- 信息密度：中高。留白用于分组，不用于装饰。

## 1. 色板（light only, v1）

| token | 值 | 用途 |
|---|---|---|
| paper | #F6F4EF | 页面底色（暖纸白） |
| surface | #FFFDF9 | 卡片 |
| surface-2 | #EFECE5 | 次级填充、代码块底 |
| ink | #17181B | 主文字 |
| ink-2 | #43464D | 正文 |
| muted | #767B85 | 辅助文字 |
| faint | #9BA0A8 | 占位、禁用 |
| line | #E2DED4 | 1px 分隔线 |
| line-strong | #CFC9BC | 强调分隔 |
| teal | #0F5C55 | 品牌主色（simple 路由） |
| indigo | #4338CA | graph 路由 |
| amber | #B45309 | agentic 路由 |
| success | #2F6B3A | 通过 |
| danger | #A3261D | 失败 / 拒绝 |

规则：同一屏最多出现 3 个语义色（teal + 一个路由色 + 一个状态色）。路由色只用于路由标识与轨迹，不用于大面积背景。

## 2. 字体

| 角色 | 栈 | 用法 |
|---|---|---|
| serif | "Noto Serif CJK SC", "Songti SC", serif | 页面标题、答案首段、数字大标题 |
| sans | "Noto Sans CJK SC", "PingFang SC", system-ui | UI 与正文 |
| mono | "DejaVu Sans Mono", "Noto Sans Mono", ui-monospace | 分数、ID、耗时、JSON |

字号：11 / 12 / 13 / 14 / 16 / 20 / 28 / 40。正文 14px / 1.65；辅助 12px；数字用 mono + tabular-nums。
字距：serif 大标题 -0.01em；mono 数字 +0.01em。禁止全大写长句（除 11px 的 eyebrow 标签）。

## 3. 间距与形状

- 4px 基准；卡片内 16/20px；区块间 24/32px；页面左右 24/32px。
- 圆角：卡片 6px，输入框 8px，chip 999px（仅状态/路由），按钮 6px。
- 边框：1px line；强调用 1px line-strong。阴影只给浮层（右侧抽屉），其余靠边框分层。

## 4. 组件约定

- **RouteBadge**：chip + 色点 + 名称 + 置信度（mono）。simple=teal / multihop=indigo / agentic=amber。
- **TraceRail**：横向 4 步（路由 → 检索 → 重排 → 生成），每步显示耗时（mono），进行中脉冲，完成后实线连接。
- **EvidenceCard**：序号（mono）+ 标题 + 来源路径 + 分数条（0-1）+ 片段；分数条用路由色，宽度=分数。
- **AnswerBody**：首段 serif 20px，其余 sans 14px；引用角标 [n] 可点击定位到证据卡；流式输出时有 2px 光标。
- **ToolTimeline**（agentic）：工具名 + 参数摘要 + 结果摘要 + 耗时，左侧竖线串起。
- **GraphPath**（multihop）：SVG 节点-边图，节点用 ink 描边，边标注关系，gold path 用 indigo。
- **StatsStrip**：延迟 / tokens / 成本 / 引用校验，mono 数字 + 11px 标签。

## 5. 状态覆盖（必须实现）

loading（骨架/脉冲）、empty（示例问题引导）、streaming（光标 + 分阶段）、success、error（可重试）、refusal（拒答样式：danger 左边框 + 原因）、partial（证据不足提示）。

## 6. 动效纪律

- 时长 120–240ms；ease-out；只动 opacity/transform。
- 允许：流式光标、轨迹脉冲、卡片进入 8px 上移淡入、分数条增长。
- 禁止：弹跳、旋转 loading 超过 1 圈、视差、自动播放。
- 尊重 `prefers-reduced-motion`：全部降级为无动画。

## 7. 无障碍底线

- 对比度 ≥ 4.5:1（正文）；焦点可见（2px teal outline + 2px offset）。
- 所有交互可键盘到达；流式回答用 `aria-live="polite"`。
- 颜色不作为唯一信息载体（路由同时有文字标签）。
- 触控目标 ≥ 40×40px。

## 8. 页面结构

顶栏 56px：字标 + 路由图例 + 系统状态。
左栏 232px：演示案例 / 历史 / 评测入口。
主栏：Composer → TraceRail → AnswerCard → StatsStrip。
右栏 400px：证据 / 图谱 / 轨迹 / 调试（Tabs）。<1200px 时右栏变抽屉。
