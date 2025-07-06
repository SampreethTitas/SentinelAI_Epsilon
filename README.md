# SentinelAI 🛡️

> Intelligent security gateway for automation tools with advanced input validation and output analysis 🛡️

[![Live Demo](https://img.shields.io/badge/Demo-Live-green?style=for-the-badge)](https://165.232.185.93/)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-blue?style=flat-square)](https://reactjs.org/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%2B%20Python-green?style=flat-square)](https://fastapi.tiangolo.com/)


## 🚀 Live Demo

**[Try SentinelAI Live](https://165.232.185.93/)**

> ⚠️ **Note**: Since this demo runs on an IP address without a domain, your browser may show a security warning. Click "Advanced options" and then "Proceed to 162.232....(unsafe)" to access the demo.

SentinelAI is a security gateway that sits between users and automation tools to provide comprehensive input validation and output analysis. GuardPrompt detects abnormal, sensitive, and malicious inputs before they reach automation platforms, while AdShield analyzes generated content for bias, PII exposure, and other security concerns.

## ✨ Features

### 🛡️ Input Validation & Output Analysis
- **Input Screening** - GuardPrompt detects abnormal, sensitive, and malicious inputs before they reach automation tools
- **Output Monitoring** - AdShield analyzes generated content for bias, PII exposure, and security violations
- **Automation Gateway** - Sophisticated security interface around existing automation platforms
- **Real-time Protection** - Continuous monitoring of inputs and outputs for security and compliance

### 🎯 Current Capabilities
- ✅ **GuardPrompt** - Filters malicious prompts, detects PII in inputs, and blocks prompt injection attacks
- ✅ **AdShield** - Analyzes AI-generated content for bias, PII exposure, harmful content, and compliance violations
- ✅ **Platform Integration** - Seamlessly wraps around existing automation tools and platforms
- ✅ **Security API** - Fast, scalable protection layer for automation workflows
- ✅ **Monitoring Dashboard** - Interface for tracking security metrics and validation results

### 🔮 Future Roadmap
- 🚧 **Data Vault** - Secure storage and management of validation results
- 🚧 **Trust Lens** - Advanced trust scoring and reputation system
- 🚧 **Enhanced Analytics** - Comprehensive security insights and reporting

## 🏗️ Architecture

### Security Gateway Flow
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│    Users    │───▶│ GuardPrompt  │───▶│ Automation  │───▶│  AdShield    │
│             │    │   (Input     │    │   Tools     │    │  (Output     │
│ Send Inputs │    │ Validation)  │    │ (Any AI/ML  │    │ Analysis)    │
└─────────────┘    └──────────────┘    │ Platform)   │    └──────────────┘
                           │            └─────────────┘            │
                           ▼                                       ▼
                   ┌──────────────┐                      ┌──────────────┐
                   │   BLOCKED    │                      │   SECURED    │
                   │ Malicious/   │                      │   Content    │
                   │ Sensitive    │                      │  Delivered   │
                   │ Inputs       │                      │              │
                   └──────────────┘                      └──────────────┘
```

### Project Structure
```
SentinelAI/
├── frontend/          # React + TypeScript monitoring dashboard
├── backend/           # FastAPI security gateway engine
└── test_agents/       # Simulated automation platforms
```

### Backend Stack
- ✅ **FastAPI** - High-performance Python web framework for security gateway APIs
- ✅ **GuardPrompt** - Input validation module that detects malicious prompts and PII
- ✅ **AdShield** - Output analysis engine that scans generated content for bias and security issues
- ✅ **Transformers** - Hugging Face ML models for content analysis and threat detection

### Frontend Stack
- ✅ **React 18** with TypeScript
- ✅ **Tailwind CSS** + **shadcn/ui** components for modern monitoring dashboard
- ✅ **Vite** for fast development and building
- ✅ **Framer Motion** for smooth status animations

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Test Agents Setup
```bash
cd test_agents
pip install -r requirements.txt
python main.py
```


## 🧪 Simulated Automation Platforms

The `test_agents` directory contains demonstration automation platforms that simulate real AI systems. These agents serve as:

- ✅ **Wrapped Platforms** - Example automation tools that SentinelAI protects and monitors
- ✅ **Testing Environment** - Allows you to see both malicious inputs being blocked and harmful outputs being detected

These agents demonstrate the complete security pipeline: GuardPrompt validates inputs before they reach the automation platforms, and AdShield analyzes any content the platforms generate.


## 📊 Security Validation Levels

| Level | Description | Action |
|-------|-------------|---------|
| ✅ **SAFE** | Content is clean and secure | Allow |
| ⚠️ **LOW** | Minor concerns, monitor | Allow with logging |
| � **MEDIUM** | Potential security risk | Block with review |
| ❌ **HIGH** | Active threat detected | Block immediately |
| 🚨 **CRITICAL** | Severe security violation | Block and alert |

## 🔗 Links

- **Live Demo**: [https://165.232.185.93/](https://165.232.185.93/)
- **Documentation**: [Coming Soon]
- **API Reference**: [Coming Soon]
- **Support**: [Create an Issue](../../issues)

## 🙏 Acknowledgments

- Hugging Face Transformers for ML model infrastructure in AI security
- FastAPI community for excellent web framework supporting security gateway APIs
- React and TypeScript teams for frontend technology powering security dashboards
- All contributors and AI security researchers

---

**⚡ Built to provide secure gateways for automation platforms worldwide** 🚧
