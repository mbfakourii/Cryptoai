from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.core import connect, latest_runs
from app.signals import latest_signals

app = FastAPI(title='Crypto AI Signal API', version='0.1.0')
STATIC = Path(__file__).resolve().parent / 'static'
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/', include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / 'index.html')

@app.get('/health')
def health():
    return {'status': 'ok', 'mode': 'research-only', 'market': 'spot', 'interval': '15m'}

@app.get('/runs')
def runs():
    return latest_runs()


@app.get('/signals')
def signals():
    return latest_signals()

@app.get('/trades/{run_id}')
def trades(run_id: int):
    with connect() as con:
        con.row_factory = __import__('sqlite3').Row
        rows = con.execute('SELECT * FROM trades WHERE run_id=? ORDER BY entry_time', (run_id,)).fetchall()
    if not rows:
        raise HTTPException(404, 'No trades found for this run')
    return [dict(row) for row in rows]
