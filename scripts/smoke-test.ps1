param(
  [string]$BaseUrl = "http://localhost:8080",
  [string]$FilePath = ""
)

if ([string]::IsNullOrWhiteSpace($FilePath)) {
  throw "Provide -FilePath to a local document."
}

$jobId = "demo-job"
$runId = "run-001"

Write-Host "1) Uploading $FilePath"
$uploadResp = curl.exe -s -X POST "$BaseUrl/v1/documents/upload" `
  -F "job_id=$jobId" `
  -F "run_id=$runId" `
  -F "file=@$FilePath"

Write-Host $uploadResp
$uploadObj = $uploadResp | ConvertFrom-Json

Write-Host "2) Asking chat question (SSE)"
$chatBody = @{
  job_id = $jobId
  run_id = $runId
  user_query = "Summarize this document and list key numbers."
  top_k = 5
  document_ids = @($uploadObj.file_id)
} | ConvertTo-Json -Depth 5

curl.exe -N -s -X POST "$BaseUrl/v1/analytics/table-line-item" `
  -H "Content-Type: application/json" `
  -d $chatBody
