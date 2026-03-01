"""
llama.cpp server wrapper for fully offline local inference.
Uses llama-server with Gemma 3 4B for text and vision tasks.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, List
import signal


class LlamaCppServer:
    """Manages llama.cpp server for local inference with Gemma 3."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        context_size: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False
    ):
        """
        Initialize llama.cpp server configuration.
        
        Args:
            model_path: Path to GGUF model
            mmproj_path: Path to vision projector
            host: Server host
            port: Server port
            context_size: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            verbose: Enable verbose logging
        """
        self.host = host
        self.port = port
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None
        
        # Find model
        models_dir = Path(os.getenv('LLAMA_CPP_MODELS', Path.home() / '.cache' / 'llama.cpp' / 'models'))
        
        if model_path:
            self.model_path = Path(model_path)
        else:
            # Look for Google official model
            self.model_path = models_dir / 'gemma-3-4b-it-q4_0.gguf'
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: ./scripts/setup-llama-cpp.sh"
            )
        
        # Check for vision projector
        if mmproj_path:
            self.mmproj_path = Path(mmproj_path)
        else:
            self.mmproj_path = models_dir / 'mmproj-model-f16-4B.gguf'
        
        # Vision support = have projector
        self.has_vision = self.mmproj_path.exists()
        self.model_type = "gemma3-vlm" if self.has_vision else "gemma3-text"
    
    def format_chat_messages(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Format messages for Gemma 3 chat template.
        
        Gemma 3 format:
        <start_of_turn>user
        {content}<end_of_turn>
        <start_of_turn>model
        """
        formatted = ""
        
        if system:
            formatted += f"<start_of_turn>system\n{system}<end_of_turn>\n"
        
        formatted += f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        return formatted
    
    def start(self, wait_ready: bool = True, timeout: int = 90):
        """Start the llama.cpp server."""
        if self.is_running():
            print(f"Server already running at http://{self.host}:{self.port}")
            return
        
        # Use local files ONLY - no HuggingFace repo
        cmd = [
            'llama-server',
            '--model', str(self.model_path),
            '--host', self.host,
            '--port', str(self.port),
            '--ctx-size', str(self.context_size),
            '--threads', str(os.cpu_count() // 2 or 4),
            '--parallel', '1',
        ]
        
        # Add mmproj explicitly if we have it
        if self.has_vision:
            cmd.extend(['--mmproj', str(self.mmproj_path)])
            if self.verbose:
                print(f"✓ Vision projector: {self.mmproj_path.name}")
        
        if self.verbose:
            print(f"Starting llama-server:")
            print(f"  {' '.join(cmd)}")
            print()
        else:
            print(f"Starting llama-server")
            if self.has_vision:
                print(f"  ✓ Vision enabled")
            print()
        
        # Start server
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Wait for server with better diagnostics
        if wait_ready:
            print("Waiting for server to be ready...")
            if not self._wait_for_ready_with_logs(timeout):
                self.stop()
                raise RuntimeError(f"Server failed to start within {timeout} seconds")

    def _wait_for_ready_with_logs(self, timeout: int = 60) -> bool:
        """Wait for server to be ready, showing logs."""
        import select
        
        start_time = time.time()
        url = f"http://{self.host}:{self.port}/health"
        
        while time.time() - start_time < timeout:
            # Check if process died
            if self.process.poll() is not None:
                print("❌ Server process died!")
                # Print any output
                stdout, _ = self.process.communicate()
                if stdout:
                    print("Server output:")
                    print(stdout[:500])
                return False
            
            # Try health check
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    print(f"✅ Server ready in {time.time() - start_time:.1f}s")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # Show server output (non-blocking)
            if self.process.stdout:
                # Check if there's data to read
                ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                if ready:
                    line = self.process.stdout.readline()
                    if line and self.verbose:
                        print(f"  [server] {line.rstrip()}")
            
            time.sleep(0.5)
        
        print(f"❌ Timeout after {timeout}s")
        return False

    def is_running(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def stop(self):
        """Stop the server."""
        if self.process is not None:
            print("Stopping llama.cpp server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            print("✓ Server stopped")
    
    def generate(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
        timeout: int = 600
    ) -> dict:
        """Generate text using Gemma 3 4B with optional image input."""
        if not self.is_running():
            raise RuntimeError("Server is not running. Call start() first.")
        
        # Gemma-specific stop tokens
        gemma_stops = ['<end_of_turn>', '<eos>', '<|endoftext|>']
        if stop:
            gemma_stops.extend(stop)
        
        # For vision, use direct completion with image token
        if image_data and self.has_vision:
            url = f"http://{self.host}:{self.port}/completion"
            
            # Simple format that works with llama.cpp
            formatted_prompt = f"<bos><start_of_turn>user\n<start_of_image>\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            payload = {
                "prompt": formatted_prompt,
                "image_data": [{"data": image_data}],
                "n_predict": min(max_tokens, 256),  # Limit for vision to avoid timeout
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stop": gemma_stops,
                "cache_prompt": False,
                "stream": False
            }
            
            if self.verbose:
                print(f"[Vision request] Image: {len(image_data)/1024:.1f} KB")
                print(f"[Vision request] Max tokens limited to {payload['n_predict']} for faster response")
        else:
            # Standard text completion
            url = f"http://{self.host}:{self.port}/completion"
            formatted_prompt = self.format_chat_messages(prompt, system)
            
            payload = {
                'prompt': formatted_prompt,
                'n_predict': max_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repeat_penalty': repeat_penalty,
                'stop': gemma_stops,
                'cache_prompt': True,
            }
            
            if self.verbose:
                print(f"[Text request] Using /completion endpoint")
        
        try:
            if self.verbose:
                print(f"[Request] Sending to {url}")
            
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract content based on endpoint
            if "/v1/chat/completions" in url:
                # Chat completions format
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens_eval = result.get("usage", {}).get("prompt_tokens", 0)
                tokens_pred = result.get("usage", {}).get("completion_tokens", 0)
            else:
                # Standard completion format
                content = result.get('content', '').strip()
                tokens_eval = result.get('tokens_evaluated', 0)
                tokens_pred = result.get('tokens_predicted', 0)
            
            # Clean up response
            for token in gemma_stops:
                content = content.replace(token, '')
            content = content.strip()
            
            return {
                'content': content,
                'tokens_evaluated': tokens_eval,
                'tokens_predicted': tokens_pred,
            }
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {timeout}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """Run llama.cpp server in standalone mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description='llama.cpp server with Gemma 3 4B')
    parser.add_argument('--model', type=str, help='Path to GGUF model')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--ctx-size', type=int, default=8192, help='Context window size')
    parser.add_argument('--n-gpu-layers', type=int, default=-1, help='GPU layers (-1 = all)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    print("Initializing llama.cpp server...")
    print()
    
    try:
        # Kill any existing llama-server first
        print("Stopping any existing llama-server processes...")
        subprocess.run(['pkill', '-f', 'llama-server'], stderr=subprocess.DEVNULL)
        time.sleep(2)
        print()
        
        server = LlamaCppServer(
            model_path=args.model,
            host=args.host,
            port=args.port,
            context_size=args.ctx_size,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose
        )
        
        print("Starting server...")
        server.start(wait_ready=True, timeout=90)
        
        print()
        print("=" * 70)
        print("✅ llama.cpp server is running!")
        print("=" * 70)
        print()
        print(f"Endpoint: http://{args.host}:{args.port}")
        print(f"Model: {server.model_path.name}")
        print(f"Type: {server.model_type}")
        
        if server.has_vision:
            print(f"Vision: ✓ Enabled ({server.mmproj_path.name})")
        else:
            print("Vision: ✗ Disabled")
        
        print()
        print("Press Ctrl+C to stop")
        print()
        
        # Set up signal handlers
        def signal_handler(sig, frame):
            print("\n\nStopping server...")
            server.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep running
        while True:
            if not server.is_running():
                print("❌ Server stopped unexpectedly!")
                break
            time.sleep(1)
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print()
        print("Run setup first:")
        print("  ./scripts/setup-llama-cpp.sh")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if 'server' in locals():
            server.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
