"""
Frontend Integration Example for AI Agent Backend
This shows how to integrate the AI Agent with React frontend
"""

# Example React component integration code (TypeScript/JavaScript)

FRONTEND_INTEGRATION_EXAMPLE = '''
// AIAgentService.ts - Service for AI Agent Backend integration

interface EmailRequest {
    prompt: string;
    recipients: string[];
    subject_hint?: string;
    use_guard_prompt?: boolean;
    use_adshield?: boolean;
}

interface EmailResponse {
    success: boolean;
    message: string;
    email_id: string;
    recipients: string[];
    subject: string;
    timestamp: string;
    guard_analysis: boolean;
    adshield_analysis: boolean;
}

interface EmailPreview {
    preview: true;
    subject: string;
    content: string;
    email_type: string;
    keywords: string[];
    metadata: any;
}

interface AnalyticsData {
    total_emails: number;
    sent_emails: number;
    failed_emails: number;
    success_rate: number;
    guard_analyses: number;
    adshield_analyses: number;
    email_types: Record<string, number>;
}

class AIAgentService {
    private baseURL = 'http://127.0.0.1:8003';

    async sendEmail(request: EmailRequest): Promise<EmailResponse> {
        const response = await fetch(`${this.baseURL}/email/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        });
        
        if (!response.ok) {
            throw new Error(`Failed to send email: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async previewEmail(prompt: string, subjectHint?: string): Promise<EmailPreview> {
        const params = new URLSearchParams({ prompt });
        if (subjectHint) params.append('subject_hint', subjectHint);
        
        const response = await fetch(`${this.baseURL}/email/preview?${params}`);
        
        if (!response.ok) {
            throw new Error(`Failed to preview email: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async getEmailLogs(limit = 50) {
        const response = await fetch(`${this.baseURL}/email/logs?limit=${limit}`);
        
        if (!response.ok) {
            throw new Error(`Failed to get email logs: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async getEmailAnalytics(): Promise<AnalyticsData> {
        const response = await fetch(`${this.baseURL}/email/analytics`);
        
        if (!response.ok) {
            throw new Error(`Failed to get analytics: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async processWithAIAgent(
        prompt: string,
        recipients?: string[],
        analyzeGuard = true,
        analyzeAdshield = true,
        sendEmail = false
    ) {
        const params = new URLSearchParams({
            prompt,
            analyze_guard: analyzeGuard.toString(),
            analyze_adshield: analyzeAdshield.toString(),
            send_email: sendEmail.toString()
        });
        
        if (recipients) {
            recipients.forEach(email => params.append('recipients', email));
        }
        
        const response = await fetch(`${this.baseURL}/ai-agent/process?${params}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to process with AI agent: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async checkHealth() {
        const response = await fetch(`${this.baseURL}/health`);
        return await response.json();
    }
}

// React Component Example
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';

const AIEmailAgent: React.FC = () => {
    const [prompt, setPrompt] = useState('');
    const [recipients, setRecipients] = useState('');
    const [preview, setPreview] = useState<EmailPreview | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
    const { toast } = useToast();
    const aiAgent = new AIAgentService();

    useEffect(() => {
        loadAnalytics();
    }, []);

    const loadAnalytics = async () => {
        try {
            const analyticsData = await aiAgent.getEmailAnalytics();
            setAnalytics(analyticsData);
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    };

    const handlePreview = async () => {
        if (!prompt.trim()) return;
        
        setIsLoading(true);
        try {
            const previewData = await aiAgent.previewEmail(prompt);
            setPreview(previewData);
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to preview email",
                variant: "destructive"
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleSendEmail = async () => {
        if (!prompt.trim() || !recipients.trim()) return;
        
        setIsLoading(true);
        try {
            const recipientList = recipients.split(',').map(email => email.trim());
            const response = await aiAgent.sendEmail({
                prompt,
                recipients: recipientList,
                use_guard_prompt: true,
                use_adshield: true
            });
            
            toast({
                title: "Email Sent! 📧",
                description: `Email sent to ${response.recipients.length} recipients`
            });
            
            // Refresh analytics
            await loadAnalytics();
            
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to send email",
                variant: "destructive"
            });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <Card className="p-6">
                <h2 className="text-2xl font-bold mb-4">🤖 AI Email Agent</h2>
                
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium mb-2">
                            Email Prompt
                        </label>
                        <Textarea
                            placeholder="Describe the email you want to send..."
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            rows={3}
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium mb-2">
                            Recipients (comma-separated)
                        </label>
                        <Input
                            placeholder="user@example.com, customer@demo.com"
                            value={recipients}
                            onChange={(e) => setRecipients(e.target.value)}
                        />
                    </div>
                    
                    <div className="flex gap-2">
                        <Button 
                            onClick={handlePreview} 
                            disabled={isLoading || !prompt.trim()}
                            variant="outline"
                        >
                            👁️ Preview
                        </Button>
                        <Button 
                            onClick={handleSendEmail} 
                            disabled={isLoading || !prompt.trim() || !recipients.trim()}
                        >
                            📧 Send Email
                        </Button>
                    </div>
                </div>
            </Card>

            {preview && (
                <Card className="p-6">
                    <h3 className="text-lg font-semibold mb-4">📧 Email Preview</h3>
                    <div className="space-y-2">
                        <p><strong>Subject:</strong> {preview.subject}</p>
                        <p><strong>Type:</strong> {preview.email_type}</p>
                        <p><strong>Keywords:</strong> {preview.keywords.join(', ')}</p>
                        <div className="mt-4">
                            <strong>Content:</strong>
                            <pre className="whitespace-pre-wrap bg-gray-100 p-4 rounded mt-2 text-sm">
                                {preview.content}
                            </pre>
                        </div>
                    </div>
                </Card>
            )}

            {analytics && (
                <Card className="p-6">
                    <h3 className="text-lg font-semibold mb-4">📊 Analytics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                            <div className="text-2xl font-bold">{analytics.total_emails}</div>
                            <div className="text-sm text-gray-600">Total Emails</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-green-600">{analytics.sent_emails}</div>
                            <div className="text-sm text-gray-600">Sent</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-red-600">{analytics.failed_emails}</div>
                            <div className="text-sm text-gray-600">Failed</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold">{analytics.success_rate.toFixed(1)}%</div>
                            <div className="text-sm text-gray-600">Success Rate</div>
                        </div>
                    </div>
                </Card>
            )}
        </div>
    );
};

export default AIEmailAgent;
'''

print("Frontend Integration Example for AI Agent Backend")
print("=" * 60)
print(FRONTEND_INTEGRATION_EXAMPLE)
