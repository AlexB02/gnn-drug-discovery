import subprocess


def log(message):
    subprocess.call(f"echo \"{message}\"", shell=True)
