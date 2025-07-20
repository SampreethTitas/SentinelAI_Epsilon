"""
Enhanced Email Service for AI Agent Backend
Provides intelligent email generation and management
"""

import re
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class AdvancedEmailService:
    """Advanced email service with intelligent content generation"""
    
    @staticmethod
    def extract_keywords(prompt: str) -> List[str]:
        """Extract key terms from the prompt for better email generation"""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', prompt.lower())
        return [word for word in words if word not in common_words]
    
    @staticmethod
    def determine_email_type(prompt: str) -> str:
        """Determine the type of email based on the prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['promotion', 'sale', 'discount', 'offer', 'deal']):
            return 'promotional'
        elif any(word in prompt_lower for word in ['newsletter', 'update', 'news', 'announcement']):
            return 'newsletter'
        elif any(word in prompt_lower for word in ['welcome', 'onboard', 'getting started']):
            return 'welcome'
        elif any(word in prompt_lower for word in ['reminder', 'due', 'expire', 'deadline']):
            return 'reminder'
        elif any(word in prompt_lower for word in ['invite', 'invitation', 'event', 'meeting']):
            return 'invitation'
        elif any(word in prompt_lower for word in ['thank', 'appreciation', 'grateful']):
            return 'thank_you'
        elif any(word in prompt_lower for word in ['alert', 'warning', 'urgent', 'important']):
            return 'alert'
        elif any(word in prompt_lower for word in ['survey', 'feedback', 'review', 'opinion']):
            return 'survey'
        else:
            return 'general'
    
    @classmethod
    def generate_subject_line(cls, prompt: str, email_type: str, subject_hint: Optional[str] = None) -> str:
        """Generate an appropriate subject line"""
        if subject_hint:
            return subject_hint
        
        keywords = cls.extract_keywords(prompt)
        
        subject_templates = {
            'promotional': [
                "🎉 Special Offer Just for You!",
                "💰 Exclusive Deal - Limited Time",
                "🔥 Don't Miss Out on This Promotion",
                "✨ Special Savings Inside"
            ],
            'newsletter': [
                "📰 Your Weekly Update",
                "📧 Latest News and Updates",
                "🗞️ This Week's Highlights",
                "📊 Monthly Roundup"
            ],
            'welcome': [
                "👋 Welcome to Our Community!",
                "🎊 Great to Have You Here!",
                "🚀 Let's Get Started Together",
                "💫 Welcome Aboard!"
            ],
            'reminder': [
                "⏰ Friendly Reminder",
                "📅 Don't Forget",
                "⚠️ Important Reminder",
                "🔔 Quick Reminder"
            ],
            'invitation': [
                "📩 You're Invited!",
                "🎪 Special Invitation",
                "📅 Join Us for",
                "🎟️ Exclusive Invitation"
            ],
            'thank_you': [
                "🙏 Thank You!",
                "💝 We Appreciate You",
                "⭐ Thanks for Being Amazing",
                "🎉 Much Appreciated"
            ],
            'alert': [
                "🚨 Important Alert",
                "⚠️ Urgent Notice",
                "📢 Important Update",
                "🔴 Action Required"
            ],
            'survey': [
                "📝 We'd Love Your Feedback",
                "🗳️ Quick Survey Request",
                "💭 Your Opinion Matters",
                "📊 Help Us Improve"
            ],
            'general': [
                "📧 Important Message",
                "💌 Message from SentinelAI",
                "📨 Quick Update",
                "✉️ Hello from Us"
            ]
        }
        
        # Select appropriate template and customize with keywords if possible
        templates = subject_templates.get(email_type, subject_templates['general'])
        base_subject = templates[0]  # Use first template as default
        
        # Try to incorporate main keywords
        if keywords:
            main_keyword = keywords[0].title()
            if email_type == 'promotional':
                base_subject = f"🎉 Special {main_keyword} Offer!"
            elif email_type == 'newsletter':
                base_subject = f"📰 {main_keyword} Updates"
            elif email_type == 'invitation':
                base_subject = f"📩 {main_keyword} Invitation"
        
        return base_subject
    
    @classmethod
    def generate_email_content(cls, prompt: str, email_type: str, keywords: List[str]) -> str:
        """Generate email content based on type and keywords"""
        
        content_templates = {
            'promotional': """
Hello!

We're excited to share an exclusive offer with you! {prompt}

🎯 What you get:
• Special pricing just for you
• Limited-time availability
• Premium quality guaranteed

Don't wait too long - this offer won't last forever!

Click here to claim your offer now!
            """,
            'newsletter': """
Hello,

Here's what's new and exciting! {prompt}

📈 This week's highlights:
• Latest updates and improvements
• Community news and events
• Tips and insights from our team

Stay tuned for more exciting updates coming your way.
            """,
            'welcome': """
Welcome to our community!

{prompt}

🚀 Here's what you can expect:
• Exclusive access to features and content
• Regular updates and newsletters
• Priority customer support
• Community events and networking

We're thrilled to have you on board and look forward to this journey together!
            """,
            'reminder': """
Hello,

This is a friendly reminder about: {prompt}

⏰ Important details:
• Please take action before the deadline
• Contact us if you need assistance
• Check your account for more information

Thank you for your attention to this matter.
            """,
            'invitation': """
You're Invited!

{prompt}

🎪 Event details:
• Date and time will be shared soon
• Special guests and activities planned
• Refreshments and networking opportunities
• RSVP required for planning purposes

We hope to see you there!
            """,
            'thank_you': """
Thank You!

{prompt}

🙏 We truly appreciate:
• Your continued support and trust
• The time you've invested with us
• Your valuable feedback and suggestions
• Being part of our community

Your support means the world to us!
            """,
            'alert': """
Important Alert

{prompt}

🚨 Action required:
• Please review the information carefully
• Take necessary steps as outlined
• Contact support if you need assistance
• Stay updated on further communications

Thank you for your immediate attention.
            """,
            'survey': """
We Value Your Opinion!

{prompt}

📝 Your feedback helps us:
• Improve our services and products
• Better understand your needs
• Create better experiences for everyone
• Make data-driven decisions

The survey takes just 2-3 minutes of your time.
            """,
            'general': """
Hello,

{prompt}

Thank you for being part of our community. We're committed to providing you with the best possible experience.

If you have any questions or need assistance, please don't hesitate to reach out to our support team.
            """
        }
        
        template = content_templates.get(email_type, content_templates['general'])
        content = template.format(prompt=prompt)
        
        # Add personalized footer
        footer = f"""

Best regards,
The SentinelAI Team

---
🤖 This email was generated by SentinelAI Epsilon AI Agent
🕒 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📧 Email Type: {email_type.replace('_', ' ').title()}
🏷️ Keywords: {', '.join(keywords[:5]) if keywords else 'N/A'}

Need help? Reply to this email or visit our support center.
        """
        
        return (content + footer).strip()
    
    @classmethod
    def generate_complete_email(cls, prompt: str, subject_hint: Optional[str] = None) -> Dict[str, str]:
        """Generate complete email with subject and content"""
        
        # Analyze the prompt
        email_type = cls.determine_email_type(prompt)
        keywords = cls.extract_keywords(prompt)
        
        # Generate subject and content
        subject = cls.generate_subject_line(prompt, email_type, subject_hint)
        content = cls.generate_email_content(prompt, email_type, keywords)
        
        logger.info(f"Generated {email_type} email with {len(keywords)} keywords")
        
        return {
            "subject": subject,
            "content": content,
            "email_type": email_type,
            "keywords": keywords,
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "prompt_length": len(prompt),
                "content_length": len(content)
            }
        }

class EmailAnalytics:
    """Email analytics and reporting"""
    
    @staticmethod
    def analyze_email_performance(email_logs: List) -> Dict:
        """Analyze email performance metrics"""
        if not email_logs:
            return {"message": "No email logs available for analysis"}
        
        total_emails = len(email_logs)
        sent_emails = sum(1 for log in email_logs if log.status == "sent")
        failed_emails = sum(1 for log in email_logs if log.status == "failed")
        
        # Email types distribution
        email_types = {}
        guard_analyses = 0
        adshield_analyses = 0
        
        for log in email_logs:
            # Count email types from metadata if available
            if hasattr(log, 'metadata') and log.metadata:
                email_type = log.metadata.get('email_type', 'unknown')
                email_types[email_type] = email_types.get(email_type, 0) + 1
            
            # Count security analyses
            if log.guard_result:
                guard_analyses += 1
            if log.adshield_result:
                adshield_analyses += 1
        
        return {
            "total_emails": total_emails,
            "sent_emails": sent_emails,
            "failed_emails": failed_emails,
            "success_rate": (sent_emails / total_emails * 100) if total_emails > 0 else 0,
            "guard_analyses": guard_analyses,
            "adshield_analyses": adshield_analyses,
            "email_types": email_types,
            "analysis_timestamp": datetime.now().isoformat()
        }
