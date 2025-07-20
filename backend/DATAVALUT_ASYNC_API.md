# DataValut Async API

The DataValut API has been updated to handle long-running database filtering operations asynchronously. This prevents HTTP timeouts and provides better user experience for large database operations.

## Changes Made

### ❌ Old Synchronous Approach
- Single endpoint: `POST /datavalut/create-filtered-db`
- Waits for entire database creation to complete
- Times out on large databases
- Blocks the HTTP connection

### ✅ New Asynchronous Approach
- **Create Job**: `POST /datavalut/create-filtered-db` - Returns job ID immediately
- **Check Status**: `GET /datavalut/job/{job_id}` - Monitor progress
- **List Jobs**: `GET /datavalut/jobs` - View all jobs
- **Cancel Job**: `DELETE /datavalut/job/{job_id}` - Cancel running job

## API Endpoints

### 1. Create Database Filtering Job

**POST** `/datavalut/create-filtered-db`

```json
{
  "source_db_connection": "postgresql://user:pass@host:port/db",
  "filter_prompt": "Extract customers from last 6 months with 3+ purchases"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Database creation job started. Use the job_id to check status.",
  "job_id": "job_a1b2c3d4e5f6",
  "status": "pending",
  "timestamp": "2025-07-20T10:30:00",
  "estimated_time": "5-30 minutes (depends on database size)"
}
```

### 2. Check Job Status

**GET** `/datavalut/job/{job_id}`

**Response:**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "running",
  "message": "Creating filtered database...",
  "progress": 0.6,
  "timestamp": "2025-07-20T10:35:00",
  "started_at": "2025-07-20T10:30:15",
  "completed_at": null,
  "result": null,
  "error": null
}
```

### 3. Job Complete Response

When `status = "completed"`:
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "completed",
  "message": "Successfully created filtered database: customer_data_filtered",
  "progress": 1.0,
  "timestamp": "2025-07-20T10:45:00",
  "started_at": "2025-07-20T10:30:15",
  "completed_at": "2025-07-20T10:45:00",
  "result": {
    "database_details": {
      "dbname": "customer_data_filtered",
      "host": "localhost",
      "port": 5432,
      "user": "filtered_user",
      "password": "***REDACTED***"
    },
    "connection_string": "postgresql://filtered_user:password@localhost:5432/customer_data_filtered",
    "database_name": "customer_data_filtered"
  },
  "error": null
}
```

### 4. List All Jobs

**GET** `/datavalut/jobs`

**Response:**
```json
{
  "success": true,
  "timestamp": "2025-07-20T10:50:00",
  "total_jobs": 3,
  "jobs": [
    {
      "job_id": "job_a1b2c3d4e5f6",
      "status": "completed",
      "message": "Successfully created filtered database",
      "timestamp": "2025-07-20T10:45:00"
    }
  ]
}
```

### 5. Cancel Job

**DELETE** `/datavalut/job/{job_id}`

**Response:**
```json
{
  "success": true,
  "message": "Job cancelled successfully",
  "job_id": "job_a1b2c3d4e5f6",
  "status": "cancelled",
  "timestamp": "2025-07-20T10:35:00"
}
```

## Job Status Values

- **pending**: Job created, waiting to start
- **running**: Job is actively processing
- **completed**: Job finished successfully
- **failed**: Job encountered an error
- **cancelled**: Job was cancelled by user

## Usage Examples

### Python Example

```python
import requests
import time

# 1. Create job
response = requests.post("http://localhost:8002/datavalut/create-filtered-db", json={
    "source_db_connection": "postgresql://user:pass@host:port/db",
    "filter_prompt": "Extract active users from last month"
})

job_id = response.json()["job_id"]
print(f"Job created: {job_id}")

# 2. Monitor progress
while True:
    status_response = requests.get(f"http://localhost:8002/datavalut/job/{job_id}")
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']} - {status_data['message']}")
    
    if status_data["status"] in ["completed", "failed", "cancelled"]:
        break
        
    time.sleep(10)  # Check every 10 seconds

# 3. Get result
if status_data["status"] == "completed":
    result = status_data["result"]
    connection_string = result["connection_string"]
    print(f"Database ready: {connection_string}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function createAndMonitorJob() {
  // 1. Create job
  const createResponse = await axios.post('http://localhost:8002/datavalut/create-filtered-db', {
    source_db_connection: 'postgresql://user:pass@host:port/db',
    filter_prompt: 'Extract active users from last month'
  });
  
  const jobId = createResponse.data.job_id;
  console.log(`Job created: ${jobId}`);
  
  // 2. Monitor progress
  while (true) {
    const statusResponse = await axios.get(`http://localhost:8002/datavalut/job/${jobId}`);
    const statusData = statusResponse.data;
    
    console.log(`Status: ${statusData.status} - ${statusData.message}`);
    
    if (['completed', 'failed', 'cancelled'].includes(statusData.status)) {
      if (statusData.status === 'completed') {
        const connectionString = statusData.result.connection_string;
        console.log(`Database ready: ${connectionString}`);
      }
      break;
    }
    
    await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds
  }
}

createAndMonitorJob();
```

### cURL Examples

```bash
# 1. Create job
curl -X POST "http://localhost:8002/datavalut/create-filtered-db" \
  -H "Content-Type: application/json" \
  -d '{
    "source_db_connection": "postgresql://user:pass@host:port/db",
    "filter_prompt": "Extract active users from last month"
  }'

# 2. Check status
curl "http://localhost:8002/datavalut/job/job_a1b2c3d4e5f6"

# 3. List all jobs
curl "http://localhost:8002/datavalut/jobs"

# 4. Cancel job
curl -X DELETE "http://localhost:8002/datavalut/job/job_a1b2c3d4e5f6"
```

## Testing

Run the test script to see the async API in action:

```bash
cd backend
python test_datavalut_async.py
```

## Benefits

1. **No Timeouts**: Jobs run in background, no HTTP connection timeout
2. **Progress Tracking**: Real-time progress updates with percentage completion
3. **Job Management**: List, monitor, and cancel jobs as needed
4. **Better UX**: Immediate response with job ID, then poll for updates
5. **Scalability**: Concurrent job processing with thread pool
6. **Error Handling**: Detailed error messages and status tracking

## Configuration

The API uses a thread pool with a maximum of 3 concurrent database operations to prevent resource exhaustion. This can be adjusted in the `main.py` file:

```python
thread_pool = ThreadPoolExecutor(max_workers=3)  # Adjust as needed
```

## Migration Guide

If you have existing code using the old synchronous API, update it to:

1. **Call the same endpoint** (`/datavalut/create-filtered-db`) but expect a job ID
2. **Poll the status endpoint** until completion
3. **Extract results** from the job status response when complete

The API maintains backward compatibility in terms of the request format and final result structure.
