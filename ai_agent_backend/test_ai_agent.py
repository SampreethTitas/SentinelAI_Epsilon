"""
Test script for AI Agent Backend
Tests all integrated services and email functionality
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any

class AIAgentTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.base_url = base_url
        
    async def test_health_check(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test health check endpoint"""
        try:
            async with session.get(f"{self.base_url}/health") as response:
                return {
                    "test": "health_check",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "health_check", "status": "error", "error": str(e), "success": False}
    
    async def test_email_preview(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test email preview functionality"""
        try:
            payload = {
                "prompt": "Send promotional email to customers from Canada about our new winter collection",
                "subject_hint": "🎿 Winter Collection Launch"
            }
            async with session.post(f"{self.base_url}/email/preview", params=payload) as response:
                return {
                    "test": "email_preview",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "email_preview", "status": "error", "error": str(e), "success": False}
    
    async def test_email_send(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test email sending functionality"""
        try:
            payload = {
                "prompt": "Welcome new customers who just signed up for our newsletter",
                "recipients": ["test@example.com", "user@demo.com"],
                "use_guard_prompt": True,
                "use_adshield": True
            }
            async with session.post(f"{self.base_url}/email/send", json=payload) as response:
                return {
                    "test": "email_send",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "email_send", "status": "error", "error": str(e), "success": False}
    
    async def test_guard_prompt(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test GuardPrompt integration"""
        try:
            payload = {
                "prompt": "Create a marketing campaign for sustainable products",
                "context": "Environmental awareness campaign"
            }
            async with session.post(f"{self.base_url}/guard-prompt/analyze", json=payload) as response:
                return {
                    "test": "guard_prompt",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "guard_prompt", "status": "error", "error": str(e), "success": False}
    
    async def test_adshield(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test AdShield integration"""
        try:
            payload = {
                "text": "Check out our amazing deals and promotions!",
                "check_type": "comprehensive"
            }
            async with session.post(f"{self.base_url}/adshield/analyze", json=payload) as response:
                return {
                    "test": "adshield",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "adshield", "status": "error", "error": str(e), "success": False}
    
    async def test_ai_agent_process(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test comprehensive AI agent processing"""
        try:
            payload = {
                "prompt": "Send a thank you email to customers who made a purchase last month",
                "recipients": ["customer@example.com"],
                "analyze_guard": True,
                "analyze_adshield": True,
                "send_email": True
            }
            async with session.post(f"{self.base_url}/ai-agent/process", params=payload) as response:
                return {
                    "test": "ai_agent_process",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "ai_agent_process", "status": "error", "error": str(e), "success": False}
    
    async def test_email_logs(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test email logs retrieval"""
        try:
            async with session.get(f"{self.base_url}/email/logs") as response:
                return {
                    "test": "email_logs",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "email_logs", "status": "error", "error": str(e), "success": False}
    
    async def test_email_analytics(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test email analytics"""
        try:
            async with session.get(f"{self.base_url}/email/analytics") as response:
                return {
                    "test": "email_analytics",
                    "status": response.status,
                    "data": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"test": "email_analytics", "status": "error", "error": str(e), "success": False}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        async with aiohttp.ClientSession() as session:
            tests = [
                self.test_health_check(session),
                self.test_email_preview(session),
                self.test_email_send(session),
                self.test_guard_prompt(session),
                self.test_adshield(session),
                self.test_ai_agent_process(session),
                self.test_email_logs(session),
                self.test_email_analytics(session)
            ]
            
            results = await asyncio.gather(*tests, return_exceptions=True)
            
            # Process results
            test_results = []
            for result in results:
                if isinstance(result, Exception):
                    test_results.append({
                        "test": "unknown",
                        "status": "exception",
                        "error": str(result),
                        "success": False
                    })
                else:
                    test_results.append(result)
            
            # Calculate summary
            total_tests = len(test_results)
            successful_tests = sum(1 for result in test_results if result.get("success", False))
            
            return {
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": total_tests - successful_tests,
                    "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
                },
                "results": test_results
            }

async def main():
    """Main test runner"""
    print("🤖 Testing AI Agent Backend...")
    print("=" * 50)
    
    tester = AIAgentTester()
    results = await tester.run_all_tests()
    
    # Print summary
    summary = results["summary"]
    print(f"\n📊 Test Summary:")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    for result in results["results"]:
        status_emoji = "✅" if result["success"] else "❌"
        print(f"{status_emoji} {result['test']}: {result['status']}")
        if not result["success"] and "error" in result:
            print(f"   Error: {result['error']}")
    
    # Print some test data if available
    print(f"\n📧 Sample Email Data:")
    for result in results["results"]:
        if result["test"] == "email_preview" and result["success"]:
            data = result["data"]
            print(f"Subject: {data.get('subject', 'N/A')}")
            print(f"Type: {data.get('email_type', 'N/A')}")
            print(f"Keywords: {', '.join(data.get('keywords', []))}")
            break

if __name__ == "__main__":
    asyncio.run(main())
