import os
import sys
import uuid
import math
import asyncio
import subprocess
import boto3
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VidForge AI", description="AWS Bedrock Video Generation API")

# Prevent aggressive browser caching of static scripts during dev
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ["/", "/index.html", "/app.js", "/style.css"]:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

PORT = int(os.environ.get('PORT', 3000))
NOVA_BUCKET = os.environ.get('NOVA_S3_BUCKET_US_EAST_1')
LUMA_BUCKET = os.environ.get('LUMA_S3_BUCKET_US_WEST_2')

MODEL_CONFIG = {
    'amazon.nova-reel-v1:0': {'region': 'us-east-1', 'bucket': NOVA_BUCKET},
    'luma.ray-v2:0': {'region': 'us-west-2', 'bucket': LUMA_BUCKET}
}

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp_jobs", exist_ok=True)

# Caching boto3 clients and active jobs
bedrock_clients = {}
s3_clients = {}
jobs = {}

def get_bedrock_client(region):
    if region not in bedrock_clients:
        bedrock_clients[region] = boto3.client('bedrock-runtime', region_name=region)
    return bedrock_clients[region]

def get_s3_client(region):
    if region not in s3_clients:
        s3_clients[region] = boto3.client('s3', region_name=region)
    return s3_clients[region]

print(f"[Config] Amazon Nova Reel -> Region: us-east-1, Bucket: {NOVA_BUCKET}")
print(f"[Config] Luma Ray -> Region: us-west-2, Bucket: {LUMA_BUCKET}")

def build_model_input(model_id: str, prompt_text: str):
    """Build model-specific payload for Bedrock async invocation."""
    if 'nova-reel' in model_id:
        return {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": prompt_text
            },
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720"
            }
        }
    else:
        return {
            "prompt": prompt_text,
            "aspect_ratio": "16:9",
            "duration": "9s"
        }

def get_clip_duration(model_id: str) -> float:
    return 6.0 if 'nova-reel' in model_id else 9.0

class GenerateRequest(BaseModel):
    prompt: str = 'A cinematic scene'
    model: str = 'amazon.nova-reel-v1:0'  # Default to Amazon Nova Reel
    duration: int = 60  # duration in seconds (e.g., 6, 30, 60, 300)

async def start_invoke_with_retry(bedrock, model_id, model_input, output_config, max_retries=15):
    loop = asyncio.get_running_loop()
    for attempt in range(max_retries):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: bedrock.start_async_invoke(
                    modelId=model_id,
                    modelInput=model_input,
                    outputDataConfig=output_config
                )
            )
            return response
        except Exception as e:
            err_str = str(e)
            if "ThrottlingException" in err_str or "Too many requests" in err_str or "429" in err_str:
                backoff = min(60, (2 ** min(attempt, 5)) + 3)
                print(f"[Throttled] AWS Bedrock active quota limit hit. Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(backoff)
            else:
                raise e
    raise Exception("AWS Bedrock rate limit exceeded after multiple retries. Please try again shortly.")

async def process_multi_clip_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return
        
    model_conf = MODEL_CONFIG.get(job['model'])
    region = model_conf['region']
    bucket = model_conf['bucket']
    bedrock = get_bedrock_client(region)
    s3 = get_s3_client(region)
    
    num_clips = len(job['clips'])
    job_dir = os.path.join("temp_jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    loop = asyncio.get_running_loop()
    
    try:
        # Process sequentially: AWS Bedrock enforces 1 active concurrent video generation per account/region
        for clip_idx in range(num_clips):
            clip = job['clips'][clip_idx]
            clip['status'] = 'Submitting'
            
            prompt_text = clip['prompt']
            model_input = build_model_input(job['model'], prompt_text)
            output_config = {'s3OutputDataConfig': {'s3Uri': f"s3://{bucket}/video-outputs/"}}
            
            print(f"[Job {job_id[:8]}] Submitting clip {clip_idx+1}/{num_clips} to Bedrock ({job['model']})...")
            response = await start_invoke_with_retry(bedrock, job['model'], model_input, output_config)
            
            arn = response.get('invocationArn')
            clip['arn'] = arn
            clip['status'] = 'InProgress'
            print(f"[Job {job_id[:8]}] Clip {clip_idx+1} ARN: {arn}")
            
            # Poll current clip until completed before starting next
            while True:
                await asyncio.sleep(6)
                status_res = await loop.run_in_executor(
                    None,
                    lambda: bedrock.get_async_invoke(invocationArn=arn)
                )
                b_status = status_res.get('status')
                if b_status == 'Completed':
                    s3_uri = status_res.get('outputDataConfig', {}).get('s3OutputDataConfig', {}).get('s3Uri', '')
                    if s3_uri.startswith('s3://'):
                        parts = s3_uri[5:].split('/', 1)
                        b_name, prefix = parts[0], parts[1]
                        
                        list_res = await loop.run_in_executor(
                            None,
                            lambda: s3.list_objects_v2(Bucket=b_name, Prefix=prefix)
                        )
                        contents = list_res.get('Contents', [])
                        video_obj = next((c for c in contents if c['Key'].endswith('.mp4')), None)
                        
                        if video_obj:
                            local_file = os.path.join(job_dir, f"clip_{clip_idx:03d}.mp4")
                            await loop.run_in_executor(
                                None,
                                lambda: s3.download_file(b_name, video_obj['Key'], local_file)
                            )
                            clip['local_path'] = local_file
                            clip['status'] = 'Completed'
                            print(f"[Job {job_id[:8]}] Clip {clip_idx+1}/{num_clips} completed & downloaded.")
                            break
                        else:
                            raise Exception("Could not find .mp4 output file in S3.")
                    else:
                        raise Exception("Invalid S3 URI returned from Bedrock.")
                elif b_status == 'Failed':
                    msg = status_res.get('failureMessage', 'Render job failed')
                    clip['status'] = 'Failed'
                    clip['error'] = msg
                    raise Exception(msg)
                    
            await asyncio.sleep(2)  # Pause between clips
            
        # Stitch using FFmpeg
        job['status'] = 'Stitching'
        print(f"[Job {job_id[:8]}] Stitching {num_clips} clips with FFmpeg...")
        
        concat_file = os.path.join(job_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for clip in job['clips']:
                abs_p = os.path.abspath(clip['local_path'])
                f.write(f"file '{abs_p}'\n")
                
        output_filename = f"{job_id}.mp4"
        output_path = os.path.abspath(os.path.join("outputs", output_filename))
        
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
            output_path
        ]
        
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            err_log = stderr.decode('utf-8', errors='ignore')
            print(f"[Job {job_id[:8]}] FFmpeg failed: {err_log[:200]}")
            job['status'] = 'Failed'
            job['error'] = "FFmpeg video stitching failed."
            return
            
        print(f"[Job {job_id[:8]}] Stitched successfully -> {output_path}")
        job['status'] = 'Completed'
        job['video_url'] = f"/outputs/{output_filename}"
        
    except Exception as exc:
        print(f"[Job {job_id[:8]}] Job execution error: {exc}")
        job['status'] = 'Failed'
        if not job.get('error'):
            job['error'] = str(exc)

@app.post("/api/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    print(f"[POST] /api/generate - Model: {request.model}, Duration: {request.duration}s")
    
    model_conf = MODEL_CONFIG.get(request.model)
    if not model_conf:
        return JSONResponse(status_code=400, content={'error': f"Model {request.model} is not configured."})
        
    if not model_conf['bucket'] or 'your-' in model_conf['bucket']:
        return JSONResponse(status_code=400, content={'error': f"S3 bucket for {model_conf['region']} is not properly configured in .env"})

    clip_dur = get_clip_duration(request.model)
    num_clips = max(1, math.ceil(request.duration / clip_dur))
    job_id = str(uuid.uuid4())
    
    clips = []
    for i in range(num_clips):
        if num_clips == 1:
            clip_prompt = request.prompt
        else:
            clip_prompt = f"{request.prompt} (Part {i+1} of {num_clips})"
            
        clips.append({
            'index': i,
            'prompt': clip_prompt,
            'status': 'Pending',
            'arn': None,
            'local_path': None,
            'error': None
        })
        
    jobs[job_id] = {
        'job_id': job_id,
        'model': request.model,
        'prompt': request.prompt,
        'duration': request.duration,
        'num_clips': num_clips,
        'status': 'InProgress',
        'error': None,
        'video_url': None,
        'clips': clips
    }
    
    background_tasks.add_task(process_multi_clip_job, job_id)
    
    return {
        'success': True,
        'jobId': job_id,
        'invocationArn': job_id,
        'numClips': num_clips,
        'region': model_conf['region']
    }

@app.get("/api/status")
async def status(job_id: str = None, arn: str = None, region: str = None):
    target_job_id = job_id or (arn if arn in jobs else None)
    
    if target_job_id:
        job = jobs.get(target_job_id)
        if not job:
            return JSONResponse(status_code=404, content={'error': 'Job not found'})
            
        completed_clips = sum(1 for c in job['clips'] if c['status'] == 'Completed')
        total_clips = len(job['clips'])
        
        if job['status'] == 'Completed':
            progress = 100
        elif job['status'] == 'Stitching':
            progress = 95
        elif job['status'] == 'Failed':
            progress = 0
        else:
            progress = int((completed_clips / total_clips) * 90) + 5
            
        return {
            'jobId': target_job_id,
            'status': job['status'],
            'completedClips': completed_clips,
            'totalClips': total_clips,
            'progress': progress,
            'videoUrl': job.get('video_url'),
            'error': job.get('error')
        }
        
    if not arn:
        return JSONResponse(status_code=400, content={'error': 'Missing jobId or arn'})
        
    if not region and arn.startswith('arn:aws:bedrock:'):
        region = arn.split(':')[3]
        
    if not region:
        return JSONResponse(status_code=400, content={'error': 'Missing region'})
        
    try:
        client = get_bedrock_client(region)
        response = client.get_async_invoke(invocationArn=arn)
        
        status_payload = {'status': response.get('status')}
        
        if response.get('status') == 'Completed':
            s3_uri = response.get('outputDataConfig', {}).get('s3OutputDataConfig', {}).get('s3Uri', '')
            if s3_uri.startswith('s3://'):
                parts = s3_uri[5:].split('/', 1)
                if len(parts) == 2:
                    bucket = parts[0]
                    prefix = parts[1]
                    
                    s3 = get_s3_client(region)
                    list_response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                    contents = list_response.get('Contents', [])
                    
                    video_obj = next((c for c in contents if c['Key'].endswith('.mp4')), None)
                    
                    if video_obj:
                        url_str = s3.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': bucket, 'Key': video_obj['Key']},
                            ExpiresIn=3600
                        )
                        status_payload['videoUrl'] = url_str
                    else:
                        status_payload['error'] = "Could not locate .mp4 file in S3 output."
                        
        elif response.get('status') == 'Failed':
            status_payload['error'] = response.get('failureMessage', 'Job failed')
            
        return status_payload
    except Exception as e:
        print(f"Status check error: {e}")
        return JSONResponse(status_code=500, content={'error': str(e)})

# Serve stitched videos from /outputs
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Serve static frontend files at root
@app.get("/")
async def root():
    return FileResponse("index.html")

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    print('')
    print('  +------------------------------------------+')
    print('  |       * VidForge AI Server Running       |')
    print('  +------------------------------------------+')
    print(f'  |  Local:  http://localhost:{PORT}            |')
    print('  |  API:    AWS Bedrock Video Generation    |')
    print('  +------------------------------------------+')
    print('')
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
