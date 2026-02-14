"""Modal app entrypoint. Run: modal run modal_app.py | modal deploy modal_app.py"""
import modal

app = modal.App("anchor")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image)
def hello() -> str:
    return "Anchor Modal workspace is ready."


@app.local_entrypoint()
def main() -> None:
    print(hello.remote())
