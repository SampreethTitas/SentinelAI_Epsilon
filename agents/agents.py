# agent_server/agents.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from schemas import AgentInfo

# Load API Key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it.")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')

# --- Marketing Agent Registry ---
AGENT_REGISTRY = [
    AgentInfo(
        id="email",
        name="Marketing Email Writer",
        description="Generates professional marketing emails for campaigns, newsletters, or announcements.",
        endpoint="/agents/generate/email",
    ),
    AgentInfo(
        id="blog",
        name="Content Marketing Blog Writer",
        description="Creates SEO-friendly and engaging blog posts to attract and retain customers.",
        endpoint="/agents/generate/blog",
    ),
    AgentInfo(
        id="social-post",
        name="Social Media Post Generator",
        description="Generates catchy posts for platforms like Twitter, LinkedIn, or Instagram.",
        endpoint="/agents/generate/social-post",
    ),
    AgentInfo(
        id="ad-copy",
        name="Ad Copy Generator",
        description="Writes high-converting headlines and descriptions for Google Ads, Facebook Ads, etc.",
        endpoint="/agents/generate/ad-copy",
    ),
]

# --- Agent Generation Functions ---

async def generate_email(prompt: str) -> str:
    full_prompt = f"""
    You are a professional marketing email copywriter.
    Based on the following instruction, write a concise and persuasive email.
    
    Instruction: "{prompt}"
    
    Email:
    """
    response = await model.generate_content_async(full_prompt)
    print(f"Generated Email: {response.text.strip()}")
    return response.text.strip()

async def generate_blog_post(prompt: str) -> str:
    full_prompt = f"""
    You are a content marketing expert who writes engaging, SEO-friendly blog posts.
    Write a blog post about the following topic. Only give the content of the blog post, introduction and conclusion nothing else.
    
    Topic: "{prompt}"
    
    Blog Post:
    """
    response = await model.generate_content_async(full_prompt)
    return response.text.strip()

async def generate_social_post(prompt: str) -> str:
    full_prompt = f"""
    You are a savvy social media manager. Create a short, engaging social media post.
    Keep it concise and include relevant hashtags.
    
    Topic/Instruction: "{prompt}"
    
    Social Media Post:
    """
    response = await model.generate_content_async(full_prompt)
    return response.text.strip()

async def generate_ad_copy(prompt: str) -> str:
    full_prompt = f"""
    You are an expert digital marketer specializing in high-converting ad copy.
    Generate 3 variations of ad copy (headline and description) for the following product or service.
    
    Product/Service Description: "{prompt}"
    
    Ad Copy Variations:
    """
    response = await model.generate_content_async(full_prompt)
    return response.text.strip()