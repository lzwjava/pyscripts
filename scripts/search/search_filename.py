import subprocess
import argparse
import sys
import os
import shutil

def check_ack():
    """Check if ack is installed."""
    if not shutil.which('ack'):
        print("Error: ack is not installed.")
        print("Please install it first:")
        print("  macOS: brew install ack")
        print("  Ubuntu/Debian: sudo apt-get install ack")
        print("  Windows: scoop install ack")
        sys.exit(1)

def search_filenames(query, ignore_case=False):
    """Search markdown filenames using ack."""
    try:
        check_ack()
        
        cmd = [shutil.which('ack')]
        if ignore_case:
            cmd.append('-i')
        
        # Search only filenames, not file contents
        cmd.append('-g')
        
        # Only search markdown files
        cmd.append('--type-add=md=.md,.markdown')
        cmd.append('--md')
        
        # Add search pattern
        cmd.append(query)
        
        # Specify directories to search
        cmd.append('original')
        
        # Execute search
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode not in [0, 1]:  # 1 means no matches found
            print("Error executing search command")
            print(result.stderr)
            return
            
        if result.stdout:
            print(result.stdout.strip())
        else:
            print("No matching filenames found")
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing search: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search markdown filenames in the repository")
    parser.add_argument("query", help="Search pattern to look for in filenames")
    parser.add_argument("-i", "--ignore-case", action="store_true", 
                      help="Case insensitive search")
    
    args = parser.parse_args()
    search_filenames(args.query, args.ignore_case)