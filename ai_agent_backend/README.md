# SentinelAI Epsilon - AI Agent Backend

A comprehensive AI Agent backend that integrates GuardPrompt, AdShield, DataValut, and Email services into a unified API.

## 🚀 Features

- **Email Agent**: Intelligent email generation and sending with natural language prompts
- **GuardPrompt Integration**: Text analysis and prompt filtering
- **AdShield Integration**: Advertisement and content protection
- **DataValut Integration**: Database filtering and management
- **Email Analytics**: Performance tracking and reporting
- **Email Preview**: Preview emails before sending
- **Comprehensive Logging**: Detailed email logs and audit trails

## 📋 API Endpoints

### Core Services
- `GET /` - Root endpoint with service information
- `GET /health` - Health check for all integrated services
- `GET /docs` - Interactive API documentation

### Email Agent
- `POST /email/send` - Send emails with AI-generated content
- `POST /email/preview` - Preview email content without sending
- `GET /email/logs` - Retrieve email logs
- `GET /email/logs/{email_id}` - Get specific email log
- `DELETE /email/logs` - Clear all email logs
- `GET /email/analytics` - Get email performance analytics

### GuardPrompt
- `POST /guard-prompt/analyze` - Analyze text using GuardPrompt

### AdShield
- `POST /adshield/analyze` - Analyze text using AdShield

### DataValut
- `POST /datavalut/create-db` - Create filtered database
- `GET /datavalut/job/{job_id}` - Get database creation job status

### AI Agent Processor
- `POST /ai-agent/process` - Comprehensive processing with multiple services

## 🛠️ Installation & Setup

1. **Navigate to the AI Agent directory:**
   ```bash
   cd ai_agent_backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   ./start_server.sh
   ```
   Or manually:
   ```bash
   python3 main.py
   ```

4. **Access the API:**
   - Server: http://127.0.0.1:8003
   - Documentation: http://127.0.0.1:8003/docs
   - Health Check: http://127.0.0.1:8003/health

## 🧪 Testing

Run the comprehensive test suite:
```bash
python3 test_ai_agent.py
```

## 📧 Email Agent Usage

### Basic Email Sending
```python
import httpx

# Send a promotional email
response = httpx.post("http://127.0.0.1:8003/email/send", json={
    "prompt": "Send promotional email to customers about our new winter collection sale",
    "recipients": ["customer@example.com"],
    "use_guard_prompt": True,
    "use_adshield": True
})
```

### Email Preview
```python
# Preview email content
response = httpx.post("http://127.0.0.1:8003/email/preview", params={
    "prompt": "Welcome new customers who just signed up",
    "subject_hint": "Welcome to SentinelAI!"
})
```

### Comprehensive AI Processing
```python
# Process with all services
response = httpx.post("http://127.0.0.1:8003/ai-agent/process", params={
    "prompt": "Send thank you email to premium customers",
    "recipients": ["premium@example.com"],
    "analyze_guard": True,
    "analyze_adshield": True,
    "send_email": True
})
```

## 📊 Email Types

The AI Agent automatically detects and generates appropriate emails for:

- **Promotional**: Sales, offers, discounts
- **Newsletter**: Updates, news, announcements
- **Welcome**: Onboarding, getting started
- **Reminder**: Due dates, deadlines, expiration
- **Invitation**: Events, meetings, invitations
- **Thank You**: Appreciation, gratitude
- **Alert**: Urgent notices, warnings
- **Survey**: Feedback requests, polls
- **General**: Default fallback type

## 🔧 Configuration

Edit `config.py` to customize:

- Server host and port
- Backend service URLs
- CORS origins
- Timeouts
- Email settings
- Feature flags

## 🏗️ Architecture

```
AI Agent Backend (Port 8003)
├── Email Agent Service
├── GuardPrompt Integration (Port 8000)
├── AdShield Integration (Port 8000)
├── DataValut Integration (Port 8002)
└── Analytics & Logging
```

## 📈 Features

### Smart Email Generation
- Automatic subject line generation
- Content type detection
- Keyword extraction
- Personalized templates

### Security Integration
- GuardPrompt analysis for prompt safety
- AdShield content verification
- Comprehensive logging

### Analytics
- Email performance metrics
- Success/failure rates
- Type distribution
- Security analysis statistics

## 🚨 Email Logging

All emails are logged with:
- Unique email ID
- Timestamp
- Recipients
- Generated content
- Security analysis results
- Delivery status
- Metadata

## 🔮 Future Enhancements

- Rate limiting
- Real email sending (SMTP)
- Advanced AI models integration
- Email templates
- Scheduled sending
- Recipient management
- A/B testing
- Advanced analytics

## 🤝 Integration with Other Services

This AI Agent backend works seamlessly with:
- Main Backend (GuardPrompt, AdShield)
- DataValut Backend
- Test Agents Backend
- Frontend React Application

## 📝 Notes

- Email sending is currently simulated (logged only)
- All services must be running for full functionality
- Email logs are stored in memory (restart clears logs)
- Supports async processing for better performance

## 🐛 Troubleshooting

1. **Service Connection Issues**: Check if backend services are running
2. **Import Errors**: Ensure all dependencies are installed
3. **Port Conflicts**: Check if port 8003 is available
4. **CORS Issues**: Verify frontend URL in CORS_ORIGINS
