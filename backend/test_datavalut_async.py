#!/usr/bin/env python3
"""
Test script for the new asynchronous DataValut API
Demonstrates how to create a database job and monitor its progress
"""

import requests
import time
import json
from typing import Dict, Any

# API Base URL
BASE_URL = "http://localhost:8002"

def create_database_job(source_db_connection: str, filter_prompt: str) -> Dict[str, Any]:
    """Create a new database filtering job"""
    url = f"{BASE_URL}/datavalut/create-filtered-db"
    
    payload = {
        "source_db_connection": source_db_connection,
        "filter_prompt": filter_prompt
    }
    
    print(f"🚀 Creating database job...")
    print(f"Filter prompt: {filter_prompt}")
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Job created successfully!")
        print(f"Job ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        print(f"Estimated time: {result.get('estimated_time', 'Unknown')}")
        return result
    else:
        print(f"❌ Failed to create job: {response.status_code}")
        print(f"Error: {response.text}")
        return {}

def check_job_status(job_id: str) -> Dict[str, Any]:
    """Check the status of a job"""
    url = f"{BASE_URL}/datavalut/job/{job_id}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get job status: {response.status_code}")
        print(f"Error: {response.text}")
        return {}

def monitor_job(job_id: str, max_wait_minutes: int = 30) -> Dict[str, Any]:
    """Monitor a job until completion or timeout"""
    print(f"\n🔍 Monitoring job {job_id}...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        status_info = check_job_status(job_id)
        
        if not status_info:
            print("❌ Failed to get job status")
            break
        
        status = status_info.get('status')
        message = status_info.get('message', '')
        progress = status_info.get('progress')
        
        # Display progress
        progress_bar = ""
        if progress is not None:
            progress_percent = int(progress * 100)
            progress_bar = f" [{progress_percent}%]"
        
        print(f"⏳ Status: {status}{progress_bar} - {message}")
        
        # Check if job is complete
        if status in ['completed', 'failed', 'cancelled']:
            print(f"\n🎯 Job finished with status: {status}")
            
            if status == 'completed':
                result = status_info.get('result', {})
                db_details = result.get('database_details', {})
                connection_string = result.get('connection_string', '')
                
                print(f"✅ Database created successfully!")
                print(f"Database name: {db_details.get('dbname', 'Unknown')}")
                print(f"Host: {db_details.get('host', 'Unknown')}")
                print(f"Port: {db_details.get('port', 'Unknown')}")
                print(f"Connection string: {connection_string}")
                
            elif status == 'failed':
                error = status_info.get('error', 'Unknown error')
                print(f"❌ Job failed: {error}")
            
            return status_info
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print(f"\n⏰ Timeout reached ({max_wait_minutes} minutes)")
            print("Job may still be running in the background")
            break
        
        # Wait before next check
        time.sleep(10)  # Check every 10 seconds
    
    return status_info

def list_all_jobs():
    """List all jobs"""
    url = f"{BASE_URL}/datavalut/jobs"
    
    print(f"\n📋 Listing all jobs...")
    
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        jobs = result.get('jobs', [])
        total = result.get('total_jobs', 0)
        
        print(f"Total jobs: {total}")
        
        if jobs:
            print("\nJobs:")
            for job in jobs[:10]:  # Show latest 10
                job_id = job.get('job_id', 'Unknown')
                status = job.get('status', 'Unknown')
                message = job.get('message', '')
                timestamp = job.get('timestamp', '')
                
                print(f"  {job_id}: {status} - {message} ({timestamp})")
        else:
            print("No jobs found")
            
        return result
    else:
        print(f"❌ Failed to list jobs: {response.status_code}")
        print(f"Error: {response.text}")
        return {}

def test_api_health():
    """Test if the API is running"""
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            health_info = response.json()
            print(f"✅ API is healthy")
            print(f"Status: {health_info.get('status', 'Unknown')}")
            print(f"Model loaded: {health_info.get('model_loaded', 'Unknown')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 DataValut Async API Test")
    print("=" * 50)
    
    # Check API health
    if not test_api_health():
        print("\n💡 Make sure the API server is running:")
        print("   cd backend && python main.py")
        return
    
    # List existing jobs
    list_all_jobs()
    
    # Example database connection and filter
    # Note: Replace with your actual database connection string
    source_db = "postgresql://user:password@localhost:5432/source_database"
    filter_prompt = "Extract all customer records from the last 6 months where the customer has made at least 3 purchases"
    
    print(f"\n🎯 Testing with sample data:")
    print(f"Source DB: {source_db}")
    print(f"Filter: {filter_prompt}")
    
    # Create job
    job_result = create_database_job(source_db, filter_prompt)
    
    if job_result and job_result.get('job_id'):
        job_id = job_result['job_id']
        
        # Monitor job progress
        final_status = monitor_job(job_id, max_wait_minutes=5)  # 5 minute timeout for demo
        
        # List jobs again to see the completed job
        print(f"\n📋 Final job list:")
        list_all_jobs()
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()
