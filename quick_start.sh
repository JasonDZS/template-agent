#!/bin/bash
# Quick Start Script for Template Report Generation Service

echo "🚀 Template Report Generation Service - Quick Start"
echo "=================================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "   $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install service dependencies
echo "📦 Installing service dependencies..."
pip install -r requirements_service.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p workdir/{templates,documents,output,downloads}
mkdir -p logs/{tasks,errors}

echo "✅ Directories created:"
echo "   📁 workdir/templates  - Template files"
echo "   📁 workdir/documents  - Knowledge base documents"
echo "   📁 workdir/output     - Generated reports"
echo "   📁 workdir/downloads  - Downloaded reports"
echo "   📁 logs              - Service logs"

# Check if template exists
if [ ! -f "workdir/template/企业信贷评估模板.md" ]; then
    echo "⚠️  No template found at workdir/template/企业信贷评估模板.md"
    echo "   You can upload templates through the API or place them manually"
fi

# Start the service
echo ""
echo "🚀 Starting the service..."
echo "   API will be available at: http://localhost:8000"
echo "   API documentation at: http://localhost:8000/docs"
echo "   Health check at: http://localhost:8000/health"
echo ""
echo "💡 Usage examples:"
echo "   # Start service: python run_service.py serve"
echo "   # Generate report: python run_service.py generate template.md"
echo "   # Check status: python run_service.py status"
echo "   # Run examples: python examples/service_usage_example.py"
echo ""
echo "🛑 Press Ctrl+C to stop the service"
echo ""

# Start the service
exec python run_service.py serve --host 0.0.0.0 --port 8000 --reload