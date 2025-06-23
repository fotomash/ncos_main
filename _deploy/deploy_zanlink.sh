#!/bin/bash
# ncOS v22.0 Zanlink - Quick Deploy Script

echo "🚀 ncOS v22.0 Zanlink Deployment"
echo "================================"

# Check for required environment variables
if [ -z "$ZANLINK_API_KEY" ]; then
    echo "❌ Error: ZANLINK_API_KEY not set"
    echo "Please run: export ZANLINK_API_KEY='your-key'"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{cache,zbar,journals,parquet,models}
mkdir -p logs
mkdir -p integrations

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "🐳 Docker detected, using Docker deployment"

    # Build images
    echo "🔨 Building Docker images..."
    docker-compose build

    # Start services
    echo "🚀 Starting services..."
    docker-compose up -d

    # Wait for services to start
    echo "⏳ Waiting for services to start..."
    sleep 10

    # Check status
    echo "✅ Services status:"
    docker-compose ps

else
    echo "🐍 Docker not found, using Python deployment"

    # Install dependencies
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt

    # Download spacy model
    echo "🧠 Downloading language model..."
    python -m spacy download en_core_web_sm

    # Start services
    echo "🚀 Starting ncOS..."
    python ncos_launcher.py &

    echo "✅ ncOS started in background"
fi

# Test the deployment
echo ""
echo "🧪 Testing deployment..."
sleep 5

# Test health endpoint
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📊 Access points:"
echo "   - Dashboard: http://localhost:8501"
echo "   - API: http://localhost:8000"
echo "   - Zanlink: https://zanlink.com/api/v1"
echo ""
echo "💡 Next steps:"
echo "   1. Configure ChatGPT with the OpenAPI schema"
echo "   2. Test with: python test_llm_integration.py"
echo "   3. Monitor logs: docker-compose logs -f"
