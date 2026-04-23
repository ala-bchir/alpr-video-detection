import modal
app = modal.App("test-app")
@app.cls(timeout=3600, scaledown_window=180)
class Test:
    @modal.method()
    def process(self): pass
