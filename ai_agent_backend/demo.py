"""
Email Agent Demonstration Script
Shows various email generation capabilities
"""

import asyncio
import httpx
import json
from datetime import datetime

class EmailDemo:
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.base_url = base_url
    
    async def demo_email_types(self):
        """Demonstrate different email types"""
        
        email_prompts = [
            {
                "type": "Promotional",
                "prompt": "Send promotional email to customers about our Black Friday sale with 50% off on all winter clothing",
                "recipients": ["customer1@example.com", "customer2@example.com"]
            },
            {
                "type": "Welcome",
                "prompt": "Welcome new customers who just signed up for our premium membership program",
                "recipients": ["newuser@example.com"]
            },
            {
                "type": "Newsletter",
                "prompt": "Send monthly newsletter with updates about new product launches and company news",
                "recipients": ["subscriber@example.com"]
            },
            {
                "type": "Reminder",
                "prompt": "Remind customers that their subscription expires in 3 days and they need to renew",
                "recipients": ["subscriber@example.com"]
            },
            {
                "type": "Thank You",
                "prompt": "Thank customers for their recent purchase and ask for feedback",
                "recipients": ["customer@example.com"]
            },
            {
                "type": "Invitation",
                "prompt": "Invite VIP customers to our exclusive product launch event next month",
                "recipients": ["vip@example.com"]
            },
            {
                "type": "Survey",
                "prompt": "Request customer feedback through a survey about their shopping experience",
                "recipients": ["customer@example.com"]
            },
            {
                "type": "Alert",
                "prompt": "Alert customers about important security update for their account",
                "recipients": ["user@example.com"]
            }
        ]
        
        print("🤖 AI Agent Email Generation Demo")
        print("=" * 50)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i, email_data in enumerate(email_prompts, 1):
                print(f"\n📧 Demo {i}: {email_data['type']} Email")
                print("-" * 30)
                
                try:
                    # Preview the email first
                    preview_response = await client.post(
                        f"{self.base_url}/email/preview",
                        params={
                            "prompt": email_data["prompt"]
                        }
                    )
                    
                    if preview_response.status_code == 200:
                        preview_data = preview_response.json()
                        print(f"Subject: {preview_data['subject']}")
                        print(f"Type: {preview_data['email_type']}")
                        print(f"Keywords: {', '.join(preview_data.get('keywords', []))}")
                        print(f"Content Preview: {preview_data['content'][:200]}...")
                        
                        # Now send the email
                        send_response = await client.post(
                            f"{self.base_url}/email/send",
                            json={
                                "prompt": email_data["prompt"],
                                "recipients": email_data["recipients"],
                                "use_guard_prompt": True,
                                "use_adshield": True
                            }
                        )
                        
                        if send_response.status_code == 200:
                            send_data = send_response.json()
                            print(f"✅ Email sent successfully!")
                            print(f"Email ID: {send_data['email_id']}")
                            print(f"Recipients: {', '.join(send_data['recipients'])}")
                        else:
                            print(f"❌ Failed to send email: {send_response.status_code}")
                    
                    else:
                        print(f"❌ Failed to preview email: {preview_response.status_code}")
                
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                # Small delay between requests
                await asyncio.sleep(1)
    
    async def demo_comprehensive_processing(self):
        """Demonstrate comprehensive AI agent processing"""
        
        print(f"\n🔄 Comprehensive AI Agent Processing Demo")
        print("=" * 50)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/ai-agent/process",
                    params={
                        "prompt": "Send promotional email about our eco-friendly product line to environmentally conscious customers",
                        "recipients": ["eco.customer@example.com"],
                        "analyze_guard": True,
                        "analyze_adshield": True,
                        "send_email": True
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Comprehensive processing completed!")
                    print(f"Timestamp: {data['timestamp']}")
                    
                    # Show analysis results
                    if 'analyses' in data:
                        if 'guard_prompt' in data['analyses']:
                            print(f"🛡️ GuardPrompt Analysis: {'✅ Completed' if not data['analyses']['guard_prompt'].get('error') else '❌ Failed'}")
                        
                        if 'adshield' in data['analyses']:
                            print(f"🔒 AdShield Analysis: {'✅ Completed' if not data['analyses']['adshield'].get('error') else '❌ Failed'}")
                    
                    if 'email' in data:
                        print(f"📧 Email Sent: {'✅ Success' if data['email'].get('success') else '❌ Failed'}")
                
                else:
                    print(f"❌ Failed: {response.status_code}")
            
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def demo_analytics(self):
        """Demonstrate email analytics"""
        
        print(f"\n📊 Email Analytics Demo")
        print("=" * 50)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Get email logs
                logs_response = await client.get(f"{self.base_url}/email/logs")
                if logs_response.status_code == 200:
                    logs_data = logs_response.json()
                    print(f"📝 Total Email Logs: {logs_data['total']}")
                
                # Get analytics
                analytics_response = await client.get(f"{self.base_url}/email/analytics")
                if analytics_response.status_code == 200:
                    analytics_data = analytics_response.json()
                    
                    if 'total_emails' in analytics_data:
                        print(f"📧 Total Emails: {analytics_data['total_emails']}")
                        print(f"✅ Sent Emails: {analytics_data['sent_emails']}")
                        print(f"❌ Failed Emails: {analytics_data['failed_emails']}")
                        print(f"📈 Success Rate: {analytics_data['success_rate']:.1f}%")
                        print(f"🛡️ GuardPrompt Analyses: {analytics_data['guard_analyses']}")
                        print(f"🔒 AdShield Analyses: {analytics_data['adshield_analyses']}")
                        
                        if analytics_data.get('email_types'):
                            print(f"📊 Email Types Distribution:")
                            for email_type, count in analytics_data['email_types'].items():
                                print(f"   {email_type}: {count}")
                    else:
                        print("📊 No analytics data available yet")
                
                else:
                    print(f"❌ Failed to get analytics: {analytics_response.status_code}")
            
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def demo_health_check(self):
        """Demonstrate health check"""
        
        print(f"\n🏥 Health Check Demo")
        print("=" * 50)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"🤖 AI Agent: {health_data.get('ai_agent', 'unknown')}")
                    
                    if 'services' in health_data:
                        print(f"🔗 Backend Services:")
                        for service, status in health_data['services'].items():
                            status_emoji = "✅" if status == "operational" else "❌" if status == "unavailable" else "⚠️"
                            print(f"   {status_emoji} {service}: {status}")
                
                else:
                    print(f"❌ Health check failed: {response.status_code}")
            
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def run_full_demo(self):
        """Run the complete demonstration"""
        print(f"🚀 Starting AI Agent Backend Demo")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Check health first
        await self.demo_health_check()
        
        # Demo different email types
        await self.demo_email_types()
        
        # Demo comprehensive processing
        await self.demo_comprehensive_processing()
        
        # Show analytics
        await self.demo_analytics()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"🌐 View API documentation at: http://127.0.0.1:8003/docs")

async def main():
    """Main demo runner"""
    demo = EmailDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    asyncio.run(main())
