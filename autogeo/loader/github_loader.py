import os
import requests
import base64
import time
from typing import Optional

def download_github_folder(
    repo_owner: str,
    repo_name: str,
    folder_path: str,
    branch: str = 'main',
    github_token: Optional[str] = None,
    delay_between_requests: float = 0.5,
    max_retries: int = 5,
    retry_delay: float = 60
) -> None:
    """Download JSON files from a GitHub repository folder.
    
    Args:
        repo_owner: GitHub repository owner username
        repo_name: GitHub repository name
        folder_path: Local folder path to save downloaded files
        branch: Git branch name (default: 'main')
        github_token: GitHub personal access token (optional, from GITHUB_TOKEN env var if not provided)
        delay_between_requests: Delay in seconds between API requests to avoid rate limiting (default: 0.5)
        max_retries: Maximum number of retries for rate limit errors (default: 5)
        retry_delay: Delay in seconds before retrying after rate limit error (default: 60)
    """
    # Get token from parameter or environment variable
    if github_token is None:
        github_token = os.getenv('GITHUB_TOKEN')
    
    # Prepare headers with token if available
    headers = {}
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/key_point?ref={branch}"
    
    # Helper function to make request with retry logic
    def make_request_with_retry(url: str, retry_count: int = 0):
        """Make a request with retry logic for rate limiting."""
        response = requests.get(url, headers=headers)
        
        # Check for rate limit error
        if response.status_code == 403:
            # Check if it's a rate limit error
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', '0')
            rate_limit_reset = response.headers.get('X-RateLimit-Reset', '0')
            
            if rate_limit_remaining == '0' and retry_count < max_retries:
                reset_time = int(rate_limit_reset)
                wait_time = max(retry_delay, reset_time - int(time.time()) + 5)
                print(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds before retry {retry_count + 1}/{max_retries}...")
                time.sleep(wait_time)
                return make_request_with_retry(url, retry_count + 1)
            else:
                response.raise_for_status()
        
        response.raise_for_status()
        return response
    
    # Get folder contents
    response = make_request_with_retry(api_url)
    contents = response.json()
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")

    # Filter JSON files
    json_files = [item for item in contents if item['type'] == 'file' and item['name'].endswith('.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        print("No JSON files found in the repository folder.")
        return
    
    # Check which files already exist
    files_to_download = []
    skipped_count = 0
    
    for item in json_files:
        file_path = os.path.join(folder_path, item['name'])
        if os.path.exists(file_path):
            skipped_count += 1
        else:
            files_to_download.append(item)
    
    print(f"Found {total_files} JSON files:")
    print(f"  - {skipped_count} already exist (skipped)")
    print(f"  - {len(files_to_download)} need to be downloaded")
    
    if len(files_to_download) == 0:
        print("All files are already downloaded. Skipping download.")
        return
    
    # Download files with rate limiting
    for idx, item in enumerate(files_to_download, 1):
        file_path = os.path.join(folder_path, item['name'])
        
        try:
            file_content_response = make_request_with_retry(item['url'])
            file_data = file_content_response.json()
            
            if 'content' in file_data:
                file_content = base64.b64decode(file_data['content'])
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                print(f"[{idx}/{len(files_to_download)}] Downloaded: {item['name']}")
            else:
                print(f"[{idx}/{len(files_to_download)}] Warning: No content found in {item['name']}")
        except Exception as e:
            print(f"[{idx}/{len(files_to_download)}] Error downloading {item['name']}: {e}")
            continue
        
        # Add delay between requests to avoid rate limiting
        if idx < len(files_to_download):
            time.sleep(delay_between_requests)
    
    print(f"\nDownload complete: {len(files_to_download)} files downloaded, {skipped_count} files skipped.")

if __name__ == "__main__":
    repo_owner = 'cxcscmu'
    repo_name = 'deepresearch_benchmarking'
    folder_path = 'key_point'
    branch = 'main'
    try:
        download_github_folder(repo_owner, repo_name, folder_path, branch)
        print("\nall .json files downloaded!")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during download: {e}")
    except Exception as e:
        print(f"An unknown error occurred: {e}")
