# Template Agent Workflow - 从文档输入到报告生成流程图

本流程图展示了基于TaskScheduler的完整工作流程，从Markdown模板输入到最终报告生成的全过程。

```mermaid
graph TD
    %% 输入阶段
    A[Markdown模板文件] --> B[MarkdownTaskSchedule]
    B --> C{解析模板结构}
    
    %% 文档解析阶段
    C --> D[parse_markdown_file_to_document_tree]
    D --> E[创建文档树结构MarkdownDocument]
    E --> F[识别标题节点HeadingNode]
    
    %% 任务生成阶段
    F --> G{分析标题层级}
    G --> H[叶子节点标题]
    G --> I[非叶子节点标题]
    
    %% 任务创建
    H --> J[创建GENERATION任务]
    I --> K[创建MERGE任务]
    
    %% 任务调度系统
    J --> L[TaskScheduler任务调度器]
    K --> L
    L --> M[注册TaskExecutor]
    
    %% 执行器注册
    M --> N[SectionAgentExecutor]
    M --> O[MergeAgentExecutor]
    
    %% 任务执行流程
    L --> P{获取就绪任务}
    P --> Q[并发执行控制max_concurrent]
    
    %% GENERATION任务执行
    Q --> R[执行GENERATION任务]
    R --> S[SectionAgent]
    S --> T[知识库检索KnowledgeRetrievalTool]
    T --> U[LLM生成内容]
    U --> V[生成章节内容]
    
    %% MERGE任务执行
    Q --> W[执行MERGE任务]
    W --> X[MergeAgent]
    X --> Y{合并模式选择}
    Y --> Z1[模型智能合并]
    Y --> Z2[简单内容拼接]
    Z1 --> AA[生成合并内容]
    Z2 --> AA
    
    %% 任务完成和依赖处理
    V --> BB[标记任务完成]
    AA --> BB
    BB --> CC[更新依赖关系]
    CC --> DD{检查所有任务状态}
    
    %% 循环执行直到完成
    DD --> |还有未完成任务| P
    DD --> |所有任务完成| EE[收集任务结果]
    
    %% 报告生成阶段
    EE --> FF[生成最终报告]
    FF --> GG[按文档顺序排序]
    GG --> HH[合并根级任务内容]
    HH --> II[保存报告文件]
    
    %% 输出
    II --> JJ[最终Markdown报告]
    
    %% 样式定义
    classDef inputNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef agentNode fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef outputNode fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef decisionNode fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    %% 应用样式
    class A,JJ inputNode
    class B,D,E,F,L,M,EE,FF,GG,HH,II processNode
    class S,X,N,O agentNode
    class JJ outputNode
    class C,G,P,Y,DD decisionNode
```

## 工作流程详细说明

### 1. 文档输入阶段
- **输入**: Markdown模板文件（如企业信贷评估模板.md）
- **解析**: 使用`parse_markdown_file_to_document_tree`解析文档结构
- **构建**: 创建包含标题层级关系的文档树

### 2. 任务生成阶段
- **叶子节点**: 没有子标题的节点创建`GENERATION`任务
- **非叶子节点**: 有子标题的节点创建`MERGE`任务
- **依赖关系**: 基于文档层级结构建立任务依赖

### 3. 任务调度阶段
- **调度器**: `TaskScheduler`管理任务执行顺序
- **并发控制**: 通过`max_concurrent`参数控制并发数
- **执行器注册**: 
  - `SectionAgentExecutor`: 处理内容生成任务
  - `MergeAgentExecutor`: 处理内容合并任务

### 4. 内容生成执行
- **SectionAgent**: 负责具体章节内容生成
  - 使用知识库检索相关信息
  - 通过LLM生成结构化内容
  - 按照模板格式要求输出

### 5. 内容合并执行
- **MergeAgent**: 负责子章节内容合并
  - **智能合并**: 使用LLM理解语义进行合并
  - **简单拼接**: 直接连接子内容（性能更好）

### 6. 报告生成阶段
- **结果收集**: 收集所有完成任务的内容
- **排序**: 按原始文档顺序排列内容
- **合并**: 组装成最终的Markdown报告
- **输出**: 保存到指定目录

## 关键特性

1. **并发执行**: 支持多任务并行处理，提高效率
2. **依赖管理**: 自动处理任务间依赖关系
3. **错误处理**: 任务失败时提供错误信息，不影响其他任务
4. **灵活配置**: 支持不同的知识库路径和合并策略
5. **进度跟踪**: 实时显示任务执行进度和统计信息

## 核心组件

- **TaskScheduler**: 任务调度核心，负责依赖解析和并发控制
- **SectionAgent**: 内容生成智能体，集成知识检索和LLM生成
- **MergeAgent**: 内容合并智能体，支持智能和简单两种合并模式
- **MarkdownTaskSchedule**: 文档解析器，将Markdown转换为任务图

---

## 时序图 - 完整处理流程

以下序列图展示了从模板输入、知识库构建到生成最终报告的详细时序过程：

```mermaid
sequenceDiagram
    participant User as 用户
    participant Main as 主程序
    participant Schedule as MarkdownTaskSchedule
    participant Parser as 文档解析器
    participant Scheduler as TaskScheduler
    participant SectionExec as SectionAgentExecutor
    participant MergeExec as MergeAgentExecutor
    participant SAgent as SectionAgent
    participant MAgent as MergeAgent
    participant KB as 知识库(ChromaDB)
    participant LLM as 大语言模型
    participant FileSystem as 文件系统

    %% 初始化阶段
    User->>Main: 启动工作流程(template_path)
    Main->>Schedule: 创建MarkdownTaskSchedule(template_path)
    
    %% 模板解析阶段
    Schedule->>Parser: parse_markdown_file_to_document_tree()
    Parser->>FileSystem: 读取模板文件
    FileSystem-->>Parser: 返回模板内容
    Parser->>Parser: 解析Markdown结构
    Parser-->>Schedule: 返回文档树MarkdownDocument
    
    %% 任务图构建
    Schedule->>Schedule: _build_task_graph()
    Schedule->>Schedule: 分析标题层级结构
    Schedule->>Schedule: 创建GENERATION任务(叶子节点)
    Schedule->>Schedule: 创建MERGE任务(非叶子节点)
    Schedule->>Schedule: 建立任务依赖关系
    Schedule-->>Main: 返回任务统计信息
    
    %% 调度器初始化
    Main->>Scheduler: 创建TaskScheduler(max_concurrent)
    Main->>SectionExec: 创建SectionAgentExecutor(knowledge_base_path)
    Main->>MergeExec: 创建MergeAgentExecutor(knowledge_base_path)
    Main->>Scheduler: register_executor(SectionExec)
    Main->>Scheduler: register_executor(MergeExec)
    Main->>Scheduler: set_global_context(report_context)
    
    %% 任务添加
    Main->>Scheduler: add_task() for each task
    
    %% 执行循环开始
    loop 任务执行轮次
        Main->>Scheduler: get_ready_tasks()
        Scheduler-->>Main: 返回就绪任务列表
        
        %% 并发执行GENERATION任务
        par GENERATION任务执行
            Main->>Scheduler: execute_task(generation_task)
            Scheduler->>SectionExec: execute(task_input)
            
            %% SectionAgent初始化
            SectionExec->>SAgent: 创建SectionAgent
            SectionExec->>SAgent: 设置section_info, report_context
            SAgent->>KB: get_knowledge_retrieval_tool()
            KB-->>SAgent: 返回知识检索工具
            
            %% 知识检索
            SectionExec->>SAgent: run_section_generation()
            SAgent->>SAgent: 构建查询语句(section_title + report_title)
            SAgent->>KB: execute(query, top_k, threshold)
            KB->>KB: 向量相似度搜索
            KB-->>SAgent: 返回相关知识内容
            
            %% LLM生成
            SAgent->>SAgent: 构建知识上下文提示
            SAgent->>LLM: ask(messages with context)
            LLM-->>SAgent: 返回生成内容
            SAgent->>SAgent: 格式化输出内容
            SAgent-->>SectionExec: 返回生成结果
            SectionExec-->>Scheduler: 返回TaskOutput
            
        and MERGE任务执行(依赖完成后)
            Main->>Scheduler: execute_task(merge_task)
            Scheduler->>MergeExec: execute(task_input)
            
            %% MergeAgent初始化
            MergeExec->>MAgent: 创建MergeAgent
            MergeExec->>MAgent: 设置section_info, child_contents
            
            %% 合并处理
            alt 智能合并模式
                MergeExec->>MAgent: run_merge() with model
                MAgent->>MAgent: 构建合并提示
                MAgent->>LLM: ask(messages with child_contents)
                LLM-->>MAgent: 返回智能合并内容
            else 简单拼接模式
                MAgent->>MAgent: 直接拼接child_contents
            end
            
            MAgent->>MAgent: 格式化合并结果
            MAgent-->>MergeExec: 返回合并内容
            MergeExec-->>Scheduler: 返回TaskOutput
        end
        
        %% 更新进度
        Scheduler->>Scheduler: 标记任务完成
        Scheduler->>Scheduler: 更新依赖关系
        Main->>Scheduler: get_progress()
        Scheduler-->>Main: 返回进度信息
    end
    
    %% 报告生成阶段
    Main->>Main: generate_final_report()
    Main->>Scheduler: get_task_results()
    Scheduler-->>Main: 返回所有任务结果
    Main->>Main: 识别根级任务
    Main->>Main: 按文档顺序排序
    Main->>Main: 组装最终报告内容
    Main->>FileSystem: 保存报告文件
    FileSystem-->>Main: 确认保存成功
    Main-->>User: 返回最终报告路径和统计信息
```

### 时序图关键说明

#### 1. 模板解析阶段 (Lines 5-15)
- 解析Markdown模板文件为文档树结构
- 识别标题层级和内容组织

#### 2. 任务图构建阶段 (Lines 16-23)
- 叶子节点创建GENERATION任务
- 非叶子节点创建MERGE任务
- 建立基于文档结构的依赖关系

#### 3. 调度器初始化 (Lines 24-33)
- 注册SectionAgentExecutor和MergeAgentExecutor
- 设置全局上下文和并发控制

#### 4. 并发任务执行 (Lines 36-73)
- **GENERATION任务**: 知识检索 → LLM生成 → 格式化输出
- **MERGE任务**: 等待依赖完成 → 智能/简单合并 → 输出结果
- 支持并发执行以提高效率

#### 5. 知识库交互 (Lines 47-52)
- 使用ChromaDB进行向量相似度搜索
- 基于查询语句检索相关文档片段
- 为LLM提供上下文信息

#### 6. 智能体协作模式
- **SectionAgent**: 专注单个章节生成，集成知识检索
- **MergeAgent**: 专注内容合并，支持智能和简单两种模式
- 每个智能体独立工作但通过TaskScheduler协调

#### 7. 最终报告生成 (Lines 75-84)
- 收集所有完成任务的结果
- 按原始文档顺序重新组织
- 生成结构化的Markdown报告