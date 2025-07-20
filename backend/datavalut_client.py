"""
DataValut Async Client
A simple Python client for the DataValut async API with WebSocket support
"""

import requests
import time
import asyncio
import websockets
import json
from typing import Dict, Any, Optional, Callable
import threading

class DataValutAsyncClient:
    """Client for DataValut Async API"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the DataValut API
        """
        self.base_url = base_url.rstrip('/')
    
    def create_job(self, source_db_connection: str, filter_prompt: str) -> Dict[str, Any]:
        """
        Create a new database filtering job
        
        Args:
            source_db_connection: Source database connection string
            filter_prompt: Natural language description of what data to include
            
        Returns:
            Job creation response containing job_id
            
        Raises:
            Exception: If job creation fails
        """
        url = f"{self.base_url}/datavalut/create-filtered-db"
        
        payload = {
            "source_db_connection": source_db_connection,
            "filter_prompt": filter_prompt
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            raise Exception(f"Failed to create job: {response.status_code} - {error_detail}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a job
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Job status information
            
        Raises:
            Exception: If job not found or API error
        """
        url = f"{self.base_url}/datavalut/job/{job_id}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise Exception(f"Job not found: {job_id}")
        else:
            error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            raise Exception(f"Failed to get job status: {response.status_code} - {error_detail}")
    
    def list_jobs(self) -> Dict[str, Any]:
        """
        List all jobs
        
        Returns:
            List of all jobs
            
        Raises:
            Exception: If API error
        """
        url = f"{self.base_url}/datavalut/jobs"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            raise Exception(f"Failed to list jobs: {response.status_code} - {error_detail}")
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a job
        
        Args:
            job_id: The job ID to cancel
            
        Returns:
            Cancellation response
            
        Raises:
            Exception: If job not found or cannot be cancelled
        """
        url = f"{self.base_url}/datavalut/job/{job_id}"
        
        response = requests.delete(url)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise Exception(f"Job not found: {job_id}")
        else:
            error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            raise Exception(f"Failed to cancel job: {response.status_code} - {error_detail}")
    
    def wait_for_completion_websocket(
        self, 
        job_id: str, 
        timeout_minutes: int = 30,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete using WebSocket for real-time updates
        
        Args:
            job_id: The job ID to wait for
            timeout_minutes: Maximum time to wait in minutes
            progress_callback: Optional callback function to call with status updates
            
        Returns:
            Final job status
            
        Raises:
            Exception: If job fails or timeout is reached
        """
        import asyncio
        import websockets
        
        async def websocket_monitor():
            ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws/job/{job_id}"
            timeout_seconds = timeout_minutes * 60
            start_time = time.time()
            
            try:
                async with websockets.connect(ws_url) as websocket:
                    async for message in websocket:
                        if time.time() - start_time > timeout_seconds:
                            raise Exception(f"Timeout reached ({timeout_minutes} minutes)")
                        
                        try:
                            data = json.loads(message)
                            
                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(data)
                            
                            if data.get('type') in ['job_status', 'job_update']:
                                status = data.get('status')
                                
                                # Check if job is complete
                                if status == 'completed':
                                    return data
                                elif status == 'failed':
                                    error = data.get('error', 'Unknown error')
                                    raise Exception(f"Job failed: {error}")
                                elif status == 'cancelled':
                                    raise Exception("Job was cancelled")
                                    
                        except json.JSONDecodeError:
                            continue  # Ignore invalid JSON
                            
            except websockets.exceptions.ConnectionClosed:
                # Fallback to HTTP polling if WebSocket fails
                return self.wait_for_completion(job_id, timeout_minutes, 10, progress_callback)
            except Exception as e:
                if "timeout" in str(e).lower():
                    raise e
                # Fallback to HTTP polling if WebSocket fails
                return self.wait_for_completion(job_id, timeout_minutes, 10, progress_callback)
        
        # Run the async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(websocket_monitor())
    
    def wait_for_completion(
        self, 
        job_id: str, 
        timeout_minutes: int = 30,
        poll_interval_seconds: int = 10,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete
        
        Args:
            job_id: The job ID to wait for
            timeout_minutes: Maximum time to wait in minutes
            poll_interval_seconds: How often to check status in seconds
            progress_callback: Optional callback function to call with status updates
            
        Returns:
            Final job status
            
        Raises:
            Exception: If job fails or timeout is reached
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while True:
            status_info = self.get_job_status(job_id)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(status_info)
            
            status = status_info.get('status')
            
            # Check if job is complete
            if status == 'completed':
                return status_info
            elif status == 'failed':
                error = status_info.get('error', 'Unknown error')
                raise Exception(f"Job failed: {error}")
            elif status == 'cancelled':
                raise Exception("Job was cancelled")
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise Exception(f"Timeout reached ({timeout_minutes} minutes)")
            
            # Wait before next check
            time.sleep(poll_interval_seconds)
    
    def create_and_wait(
        self,
        source_db_connection: str,
        filter_prompt: str,
        timeout_minutes: int = 30,
        poll_interval_seconds: int = 10,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        use_websocket: bool = True
    ) -> Dict[str, Any]:
        """
        Create a job and wait for completion (convenience method)
        
        Args:
            source_db_connection: Source database connection string
            filter_prompt: Natural language description of what data to include
            timeout_minutes: Maximum time to wait in minutes
            poll_interval_seconds: How often to check status in seconds (HTTP only)
            progress_callback: Optional callback function to call with status updates
            use_websocket: Whether to use WebSocket for real-time updates (default: True)
            
        Returns:
            Job result containing database details and connection string
            
        Raises:
            Exception: If job creation or execution fails
        """
        # Create job
        job_response = self.create_job(source_db_connection, filter_prompt)
        job_id = job_response['job_id']
        
        # Wait for completion
        if use_websocket:
            try:
                final_status = self.wait_for_completion_websocket(
                    job_id, 
                    timeout_minutes, 
                    progress_callback
                )
            except Exception as e:
                # Fallback to HTTP polling if WebSocket fails
                final_status = self.wait_for_completion(
                    job_id, 
                    timeout_minutes, 
                    poll_interval_seconds, 
                    progress_callback
                )
        else:
            final_status = self.wait_for_completion(
                job_id, 
                timeout_minutes, 
                poll_interval_seconds, 
                progress_callback
            )
        
        # Return the result
        return final_status.get('result', {})
    
    def health_check(self) -> bool:
        """
        Check if the API is healthy
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

# Convenience functions
def create_filtered_database(
    source_db_connection: str,
    filter_prompt: str,
    base_url: str = "http://localhost:8002",
    timeout_minutes: int = 30,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Create a filtered database (high-level convenience function)
    
    Args:
        source_db_connection: Source database connection string
        filter_prompt: Natural language description of what data to include
        base_url: Base URL of the DataValut API
        timeout_minutes: Maximum time to wait in minutes
        progress_callback: Optional callback function to call with status updates
        
    Returns:
        Database details and connection string
        
    Example:
        >>> def on_progress(status):
        ...     print(f"Progress: {status.get('progress', 0)*100:.1f}% - {status.get('message', '')}")
        ...
        >>> result = create_filtered_database(
        ...     "postgresql://user:pass@host:5432/source_db",
        ...     "Extract customers from last 6 months with 3+ purchases",
        ...     progress_callback=on_progress
        ... )
        >>> print(f"New database: {result['connection_string']}")
    """
    client = DataValutAsyncClient(base_url)
    return client.create_and_wait(
        source_db_connection,
        filter_prompt,
        timeout_minutes,
        10,  # 10 second poll interval
        progress_callback
    )

# Example usage
if __name__ == "__main__":
    # Simple progress callback
    def show_progress(status_info):
        progress = status_info.get('progress', 0) * 100
        message = status_info.get('message', '')
        print(f"Progress: {progress:.1f}% - {message}")
    
    try:
        # Example usage
        result = create_filtered_database(
            source_db_connection="postgresql://user:password@localhost:5432/source_database",
            filter_prompt="Extract all customer records from the last 6 months where the customer has made at least 3 purchases",
            timeout_minutes=5,  # 5 minute timeout for demo
            progress_callback=show_progress
        )
        
        print(f"\n✅ Database created successfully!")
        print(f"Connection string: {result.get('connection_string')}")
        print(f"Database name: {result.get('database_name')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
