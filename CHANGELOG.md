# v0.1.0

## 🎉 Initial Release

Template Agent Workflow v0.1.0 - A comprehensive intelligent report generation system built on modular agent architecture.

### ✨ Features

#### 🤖 Agent Framework
- **Multi-Agent Architecture**: BaseAgent, ReActAgent, and ToolCallAgent with standardized interfaces
- **Report Generation Agents**: ReportGeneratorAgent and SectionAgent for template-based content creation
- **ReAct Pattern Implementation**: Reasoning + Acting workflow with think() and act() methods
- **State Management**: Defined agent states (IDLE, RUNNING, FINISHED, ERROR) with safe transitions

#### 🔧 Tool System  
- **Extensible Tool Framework**: BaseTool abstract class with standardized interface
- **Knowledge Retrieval**: Semantic search using ChromaDB and OpenAI-compatible embeddings
- **LLM Integration**: CreateChatCompletion tool with type conversion and JSON schema generation
- **Tool Collection Management**: Container class for organizing and executing multiple tools

#### 📄 Document Processing
- **Bidirectional Conversion**: Markdown ↔ JSON conversion with metadata preservation
- **Multi-format Support**: Handles .md, .txt, and .json documents
- **Structured Parsing**: Support for headers, lists, tables, code blocks, and images
- **Template System**: Comment-based section definitions for report generation

#### 🗄️ Knowledge Base
- **Vector Database**: ChromaDB integration with persistent storage
- **Semantic Search**: OpenAI-compatible embedding function with multiple provider support
- **Automatic Indexing**: Document collection processing with duplicate detection
- **Knowledge Retrieval**: Similarity-based document search with configurable thresholds

#### ⚙️ Configuration & CLI
- **Environment Configuration**: Full .env support for API keys and settings
- **Command Line Interface**: Four core operations (generate, convert, test, list)
- **Multi-provider Support**: OpenAI, HuggingFace, Ollama embedding providers
- **Flexible Settings**: Configurable model parameters and processing options

#### 📊 Report Generation
- **Template-based Workflow**: Generate reports from Markdown templates
- **Knowledge Integration**: Automatic retrieval of relevant information
- **Parallel Processing**: Concurrent section generation for improved performance
- **Progress Tracking**: Real-time monitoring of generation progress
- **Multi-format Output**: Both JSON and Markdown report formats

### 🛠️ Technical Highlights
- **Robust Error Handling**: Comprehensive error management across all components  
- **Resource Management**: Automatic cleanup of database connections and resources
- **Type Safety**: Pydantic-based models with strong typing throughout
- **Extensible Architecture**: Plugin-ready design for custom tools and agents
- **Performance Optimization**: Parallel processing and efficient vector operations

### 📦 Components
- **Core Modules**: agent, tool, schema, converter, config, llm, logger
- **Agent Types**: BaseAgent, ReActAgent, ToolCallAgent, ReportGeneratorAgent, SectionAgent
- **Tools**: CreateChatCompletion, KnowledgeRetrievalTool, Terminate
- **Utilities**: MarkdownConverter, MarkdownParser, MarkdownRenderer
- **CLI Interface**: Main entry point with comprehensive command handling

### 🚀 Getting Started
- Install dependencies with `pip install -r requirements.txt`
- Configure environment variables in `.env` file
- Use CLI commands: `python main.py {generate|convert|test|list}`
- Place documents in `workdir/documents/` for knowledge base
- Create templates in `workdir/template/` for report generation
