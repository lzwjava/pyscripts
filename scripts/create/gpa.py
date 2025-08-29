import os
import subprocess
import platform
import sys

def gpa():
    python_exec = sys.executable  # full path to current Python interpreter

    system = platform.system()
    if system == "Linux":
        shell_command = f"bash -l -c '{python_exec} ~/bin/gitmessageai.py --api deepseek --allow-pull-push'"
    elif system == "Darwin":
        shell_command = f"zsh -l -c '{python_exec} ~/bin/gitmessageai.py --api deepseek --allow-pull-push'"
    else:  # Windows
        shell_command = f'cmd.exe /c "{python_exec} %USERPROFILE%\\bin\\gitmessageai.py --api deepseek --allow-pull-push"'

    subprocess.run(shell_command, shell=True)
