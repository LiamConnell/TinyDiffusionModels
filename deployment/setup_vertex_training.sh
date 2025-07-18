#!/bin/bash
# Setup script for Vertex AI training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up Vertex AI training environment${NC}"

# Check if required tools are installed
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        exit 1
    fi
}

echo "🔍 Checking required tools..."
check_tool "gcloud"
check_tool "docker"
check_tool "python3"

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Please authenticate with gcloud first:${NC}"
    echo "gcloud auth login"
    exit 1
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ No default project set. Please set one:${NC}"
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✅ Using project: $PROJECT_ID${NC}"

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable compute.googleapis.com --project=$PROJECT_ID

# Create a bucket for outputs (if it doesn't exist)
BUCKET_NAME="${PROJECT_ID}-text-diffusion"
if ! gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    echo "🪣 Creating GCS bucket: $BUCKET_NAME"
    gsutil mb gs://$BUCKET_NAME
else
    echo -e "${GREEN}✅ Bucket gs://$BUCKET_NAME already exists${NC}"
fi

# Make the submission scripts executable
chmod +x deployment/submit_vertex_training.py
chmod +x deployment/submit_vertex_sampling.py

# Create .env file if it doesn't exist
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    
    # Update with detected values
    sed -i "s/your-gcp-project-id/$PROJECT_ID/g" .env
    sed -i "s/your-bucket-name/$BUCKET_NAME/g" .env
    
    echo -e "${YELLOW}⚠️  Please edit .env file and add your HF_TOKEN${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your HF_TOKEN:"
echo -e "${YELLOW}   HF_TOKEN=your_huggingface_token${NC}"
echo ""
echo "2. Submit a training job:"
echo -e "${YELLOW}   python deployment/submit_vertex_training.py${NC}"
echo ""
echo "   Or override specific settings:"
echo -e "${YELLOW}   python deployment/submit_vertex_training.py --epochs 5 --batch-size 32${NC}"
echo ""
echo "3. Submit a sampling job (after training):"
echo -e "${YELLOW}   python deployment/submit_vertex_sampling.py${NC}"
echo ""
echo "   Or with specific settings:"
echo -e "${YELLOW}   python deployment/submit_vertex_sampling.py --num-samples 10 --sample-mode guided_sample${NC}"
echo ""
echo "Configuration is now managed through .env file!"
echo "See .env.example for all available options."