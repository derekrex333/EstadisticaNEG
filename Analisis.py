"""Beta bancaria: SPY (mercado) vs XLF (bancos), 2010-2025, datos mensuales."""
import json, numpy as np, pandas as pd, yfinance as yf
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.stattools import durbin_watson

px = yf.download(["SPY", "XLF"], start="2009-12-01", end="2026-01-31", interval="1mo",
                 auto_adjust=True, progress=False)["Close"]
px.index = px.index.to_period("M")
print("Filas descargadas:", len(px), "| columnas:", list(px.columns))
df = px.rename(columns={"SPY": "spy", "XLF": "xlf"}).dropna().loc["2010-01":"2025-12"]

def modelo(x, y):
    m = sm.OLS(y, sm.add_constant(x)).fit()
    r, p = pearsonr(x, y)
    res = np.asarray(m.resid)
    return dict(n=int(len(x)), r=r, p=p, b0=m.params[0], b1=m.params[1], pb1=m.pvalues[1],
                r2=m.rsquared, dw=durbin_watson(res),
                ac_riesgo=np.corrcoef(res[:-1] ** 2, res[1:] ** 2)[0, 1],
                x=list(x), y=list(y), fit=list(m.fittedvalues), res=list(res))

niv = modelo(df.spy.values, df.xlf.values)
d = (df.pct_change() * 100).dropna()
cam = modelo(d.spy.values, d.xlf.values)
out = dict(fechas=[str(i) for i in df.index], fechas_cam=[str(i) for i in d.index], niveles=niv, cambios=cam)
clean = lambda o: {k: clean(v) for k, v in o.items()} if isinstance(o, dict) else \
    [float(a) for a in o] if isinstance(o, list) and o and not isinstance(o[0], str) else o
with open("results.js", "w") as f:
    f.write("window.RESULTS=" + json.dumps(clean(out)) + ";")

N, C = niv, cam
confiable = "no son confiables" if N["dw"] < 1 else "hay que leerlos con cuidado"
perfil = "más volátil que el mercado" if C["b1"] > 1 else "menos volátil que el mercado"

print(f"NIVELES: r={N['r']:.3f} R2={N['r2']:.3f} DW={N['dw']:.2f} | CAMBIOS: r={C['r']:.3f} beta={C['b1']:.3f} R2={C['r2']:.3f} DW={C['dw']:.2f}")
print("Listo: abre index.html y lee guion.md")