import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from xml.sax.saxutils import escape
from datetime import datetime

st.set_page_config(page_title="EDEAS Six Sigma AI Lab", layout="wide")
st.title("EDEAS | Six Sigma AI Lab")
st.caption("Training Edition · Exploratory Data Analysis · Distribution Fit · Capability Analysis · Stability Analysis")

@st.cache_data
def demo_data():
    return pd.read_csv(__import__("pathlib").Path(__file__).resolve().parent / "data" / "thickness.csv")

@st.cache_data(show_spinner=False)
def read_uploaded(raw, filename, sheet=None):
    bio = io.BytesIO(raw)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio, sheet_name=sheet if sheet is not None else 0)
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Codifica del file non riconosciuta")

def numeric_values(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    cleaned = series.astype("string").str.strip().str.replace(" ", "", regex=False)
    cleaned = cleaned.map(lambda v: v.replace(",", ".") if isinstance(v, str) and "," in v and "." not in v else v)
    return pd.to_numeric(cleaned, errors="coerce").to_numpy(dtype=float, na_value=np.nan)

def summary(x):
    n = len(x)
    if n < 8 or np.std(x, ddof=1) == 0:
        raise ValueError("Servono almeno 8 osservazioni valide con variabilità non nulla.")
    mean, sd = np.mean(x), np.std(x, ddof=1)
    ad, p = normal_ad(x)
    ci_mean = stats.t.interval(.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
    chi_lo, chi_hi = stats.chi2.ppf([.025, .975], n - 1)
    ci_sd = (np.sqrt((n - 1) * sd**2 / chi_hi), np.sqrt((n - 1) * sd**2 / chi_lo))
    k = 0
    for candidate in range(1, n // 2 + 1):
        coverage = 1 - 2 * stats.binom.cdf(candidate - 1, n, .5)
        if coverage >= .95:
            k = candidate
        else:
            break
    sx = np.sort(x)
    ci_median = (sx[k - 1], sx[n - k]) if k else (sx[0], sx[-1])
    return dict(n=n, mean=mean, sd=sd, var=sd**2,
                skew=stats.skew(x, bias=False), kurt=stats.kurtosis(x, bias=False),
                ad=ad, p=p, q=np.quantile(x, [0, .25, .5, .75, 1]),
                ci_mean=ci_mean, ci_sd=ci_sd, ci_median=ci_median)

def summary_figure(x, r, variable):
    fig = plt.figure(figsize=(14, 7.875), facecolor="#eeeeee")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[3, 1],
                          left=.07, right=.96, top=.88, bottom=.12, wspace=.20, hspace=.24)
    ax = fig.add_subplot(gs[0, 0]); bx = fig.add_subplot(gs[1, 0], sharex=ax)
    tx = fig.add_subplot(gs[:, 1]); tx.axis("off")
    fig.suptitle(f"Summary Report for {variable}", fontsize=19, fontweight="bold", y=.965)
    _, edges, _ = ax.hist(x, bins="auto", color="#8bc7e8", edgecolor="white")
    xx = np.linspace(min(edges[0], min(x)), max(edges[-1], max(x)), 500)
    ax.plot(xx, stats.norm.pdf(xx, r["mean"], r["sd"]) * len(x) * np.mean(np.diff(edges)), color="#b72e34", lw=2.3)
    ax.set_ylabel("Frequency"); ax.set_facecolor("white")
    bx.boxplot(x, vert=False, patch_artist=True,
               boxprops={"facecolor": "#8bc7e8", "edgecolor": "#4d7591"},
               medianprops={"color": "#a42a31", "linewidth": 2},
               flierprops={"marker": "o", "markersize": 5})
    bx.set_yticks([]); bx.set_xlabel(variable); bx.set_facecolor("white")
    def line(y): tx.plot([.03, .98], [y, y], transform=tx.transAxes, color="#aaa", lw=.8)
    def heading(y, s): tx.text(.04, y, s, transform=tx.transAxes, fontsize=12, fontweight="bold", va="top")
    def item(y, k, v):
        tx.text(.05, y, k, transform=tx.transAxes, fontsize=10, va="top")
        tx.text(.96, y, str(v), transform=tx.transAxes, fontsize=10, va="top", ha="right")
    f = lambda v: f"{v:.4f}"
    heading(.98, "Anderson-Darling Normality Test")
    item(.93, "A-Squared", f(r["ad"])); item(.89, "P-Value", f(r["p"])); line(.85)
    heading(.83, "Descriptive Statistics")
    for i, (k, v) in enumerate([("Mean", r["mean"]), ("StDev", r["sd"]), ("Variance", r["var"]),
                                 ("Skewness", r["skew"]), ("Kurtosis", r["kurt"]), ("N", r["n"])]):
        item(.78 - i * .043, k, str(v) if k == "N" else f(v))
    line(.50); heading(.48, "Five-Number Summary")
    for i, (k, v) in enumerate(zip(["Minimum", "1st Quartile", "Median", "3rd Quartile", "Maximum"], r["q"])):
        item(.43 - i * .043, k, f(v))
    line(.20); heading(.18, "95% Confidence Intervals")
    for i, (k, key) in enumerate([("Mean", "ci_mean"), ("Median", "ci_median"), ("StDev", "ci_sd")]):
        a, b = r[key]; item(.135 - i * .043, k, f"({a:.3f}, {b:.3f})")
    return fig

def c4(n):
    """Bias correction for sample standard deviation of n normal observations."""
    from scipy.special import gammaln
    return float(np.exp(0.5 * np.log(2 / (n - 1)) + gammaln(n / 2) - gammaln((n - 1) / 2)))

D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
      7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
D3 = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848,
      7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}

def subgroup_statistics(groups):
    """Return within estimate and subgroup statistics, preserving file order."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    sizes = np.array([len(g) for g in groups])
    if len(groups) < 2 or np.any(sizes < 2):
        return None, 'Occorrono almeno 2 sottogruppi, ciascuno con almeno 2 osservazioni valide.'
    means = np.array([g.mean() for g in groups])
    stds = np.array([g.std(ddof=1) for g in groups])
    ranges = np.array([np.ptp(g) for g in groups])
    equal = bool(np.all(sizes == sizes[0]))
    n = int(sizes[0]) if equal else None
    if equal and 2 <= n <= 10:
        sigma = float(ranges.mean() / D2[n])
        method = f'R̄/d₂ (n = {n}; d₂ = {D2[n]})'
    else:
        df = int(np.sum(sizes - 1))
        pooled = float(np.sqrt(np.sum((sizes - 1) * stds**2) / df))
        sigma = pooled / c4(df + 1)
        method = f'Sp/c₄ (df = {df}; correzione c₄({df + 1}))'
    return dict(groups=groups, sizes=sizes, means=means, stds=stds, ranges=ranges,
                equal=equal, n=n, sigma=sigma, method=method), None

def subgroup_charts(info, labels, variable):
    """Phase I Xbar-R or Xbar-S charts for equal-sized rational subgroups."""
    if not info['equal']:
        return None, 'Carte X̄-R / X̄-S non disponibili con numerosità variabile: usa sottogruppi di uguale dimensione.'
    n = info['n']
    if n < 2:
        return None, 'Dimensione sottogruppo non valida.'
    means, ranges, stds = info['means'], info['ranges'], info['stds']
    center = float(np.mean(means))
    if 2 <= n <= 10:
        rbar = float(np.mean(ranges))
        a2 = 3 / (D2[n] * np.sqrt(n))
        ucl_x, lcl_x = center + a2 * rbar, center - a2 * rbar
        ucl_v = (1 + 3 * D3[n] / D2[n]) * rbar
        lcl_v = max(0, (1 - 3 * D3[n] / D2[n]) * rbar)
        values, vcenter, label = ranges, rbar, 'R'
    else:
        sbar = float(np.mean(stds))
        cn = c4(n)
        a3 = 3 / (cn * np.sqrt(n))
        ucl_x, lcl_x = center + a3 * sbar, center - a3 * sbar
        b3 = max(0, 1 - 3 * np.sqrt(1 - cn**2) / cn)
        b4 = 1 + 3 * np.sqrt(1 - cn**2) / cn
        ucl_v, lcl_v = b4 * sbar, b3 * sbar
        values, vcenter, label = stds, sbar, 'S'
    xx = np.arange(1, len(means) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    for ax, yy, cl, lo, hi, title in [
        (ax1, means, center, lcl_x, ucl_x, 'X̄ Chart'),
        (ax2, values, vcenter, lcl_v, ucl_v, f'{label} Chart')]:
        ax.plot(xx, yy, 'o-', color='#287ca5', markersize=4)
        ax.axhline(cl, color='green', label='CL')
        ax.axhline(hi, color='#b72e34', linestyle='--', label='UCL / LCL')
        ax.axhline(lo, color='#b72e34', linestyle='--')
        bad = (yy > hi) | (yy < lo)
        if np.any(bad): ax.scatter(xx[bad], yy[bad], color='red', s=60, zorder=5)
        ax.set_title(title); ax.set_ylabel(variable if ax is ax1 else label)
        ax.legend(loc='best')
    ax2.set_xlabel('Sottogruppo (ordine nel file)')
    fig.tight_layout()
    count = int(np.sum((means > ucl_x) | (means < lcl_x)) +
                np.sum((values > ucl_v) | (values < lcl_v)))
    return (fig, count, label), None

def show_png(fig, name):
    st.pyplot(fig, use_container_width=True)
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Scarica grafico PNG", b.getvalue(), file_name=name, mime="image/png")
    plt.close(fig)


class MixtureDistribution:
    """Univariate Gaussian mixture with scipy-like pdf/cdf/ppf interface."""
    def __init__(self, weights, means, stds):
        self.weights = np.asarray(weights, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)
    def pdf(self, z):
        z = np.asarray(z)
        return sum(w * stats.norm.pdf(z, m, sd) for w, m, sd in zip(self.weights, self.means, self.stds))
    def cdf(self, z):
        z = np.asarray(z)
        return sum(w * stats.norm.cdf(z, m, sd) for w, m, sd in zip(self.weights, self.means, self.stds))
    def ppf(self, q):
        low = float(np.min(self.means - 12 * self.stds))
        high = float(np.max(self.means + 12 * self.stds))
        return float(brentq(lambda z: float(self.cdf(z)) - q, low, high))

def fit_models(x, max_components=3, include_mixtures=True):
    models, rows = {}, []
    for label, dist, kw in [("Normal", stats.norm, {}),
                            ("Lognormal", stats.lognorm, {"floc": 0}),
                            ("Weibull", stats.weibull_min, {"floc": 0}),
                            ("Gamma", stats.gamma, {"floc": 0})]:
        if kw and np.any(x <= 0):
            continue
        try:
            pars = dist.fit(x, **kw)
            ll = float(np.sum(dist.logpdf(x, *pars)))
            if not np.isfinite(ll): continue
            k = len(pars) - len(kw)
            model = {"label": label, "pdf": lambda z, d=dist, p=pars: d.pdf(z, *p),
                     "cdf": lambda z, d=dist, p=pars: d.cdf(z, *p),
                     "ppf": lambda q, d=dist, p=pars: d.ppf(q, *p),
                     "parameters": ', '.join(f'{v:.5g}' for v in pars)}
            models[label] = model
            rows.append({"Distribuzione": label, "AIC": 2*k-2*ll,
                         "BIC": k*np.log(len(x))-2*ll, "Parametri": model['parameters']})
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if include_mixtures and len(x) >= 20:
        for n in range(2, max_components + 1):
            try:
                gm = GaussianMixture(n_components=n, covariance_type='full',
                                     n_init=5, reg_covar=1e-6, random_state=42,
                                     max_iter=500).fit(x.reshape(-1, 1))
                weights = gm.weights_.copy()
                means = gm.means_.reshape(-1).copy()
                stds = np.sqrt(gm.covariances_.reshape(-1)).copy()
                if not gm.converged_: continue
                md = MixtureDistribution(weights, means, stds)
                label = f'GMM {n} componenti'
                models[label] = {"label": label, "pdf": md.pdf, "cdf": md.cdf,
                                 "ppf": md.ppf,
                                 "parameters": '; '.join(f'w={w:.3f}, μ={m:.4g}, σ={sd:.4g}'
                                                        for w,m,sd in zip(weights, means, stds))}
                rows.append({"Distribuzione": label, "AIC": float(gm.aic(x.reshape(-1, 1))),
                             "BIC": float(gm.bic(x.reshape(-1, 1))),
                             "Parametri": models[label]['parameters']})
            except (ValueError, RuntimeError, FloatingPointError):
                continue
    return models, pd.DataFrame(rows).sort_values('BIC').reset_index(drop=True)

def make_pdf_report(x, all_x, r, variable, fit_table, models, selected_model,
                    lsl, usl, use_lsl, use_usl, sigma_within, within_method,
                    sg, selected_charts, strat_series=None):
    """Create a self-contained, on-demand PDF using the current session inputs."""
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=(21*cm, 29.7*cm), rightMargin=1.7*cm,
                            leftMargin=1.7*cm, topMargin=1.6*cm, bottomMargin=1.6*cm)
    styles = getSampleStyleSheet()
    styles['Title'].textColor = colors.HexColor('#17577e')
    story = []
    def para(t, style='Normal'):
        story.append(Paragraph(escape(str(t)), styles[style]))
        story.append(Spacer(1, 0.17*cm))
    def heading(t):
        story.append(Spacer(1, .25*cm)); para(t, 'Heading2')
    def table(rows):
        rows = [[str(c) for c in row] for row in rows]
        t = Table([[Paragraph(escape(v), styles['BodyText']) for v in row] for row in rows],
                  repeatRows=1, hAlign='LEFT', colWidths=[(17.4/len(rows[0]))*cm]*len(rows[0]))
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eef4')),
            ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c4d4dc')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6)]))
        story.append(t); story.append(Spacer(1,.25*cm))
    def addfig(fig):
        b=io.BytesIO(); fig.savefig(b,format='png',dpi=140,bbox_inches='tight'); plt.close(fig)
        b.seek(0)
        from PIL import Image as PILImage
        im=PILImage.open(b); w,h=im.size; b.seek(0)
        width=min(17.2*cm, w/140*2.54*cm)
        height=width*h/w
        if height>15*cm: height=15*cm; width=height*w/h
        story.append(Image(b,width=width,height=height)); story.append(Spacer(1,.2*cm))
    para('EDEAS | Six Sigma AI Lab', 'Title')
    para('Statistical analysis report - Training Edition', 'Heading2')
    para(f'Generated: {datetime.now():%Y-%m-%d %H:%M} | Variable: {variable} | Valid observations: {len(x)}')
    para('Interpret results only after checking measurement quality, sampling, time order and process stability.')
    heading('1. Exploratory Data Analysis')
    table([['Statistic','Value'],['N',len(x)],['Mean',f'{r["mean"]:.5g}'],
           ['StDev overall',f'{r["sd"]:.5g}'],['Median',f'{np.median(x):.5g}'],
           ['Minimum / Maximum',f'{np.min(x):.5g} / {np.max(x):.5g}'],
           ['Anderson-Darling normality p-value',f'{r["p"]:.4g}']])
    para('Normality test: p < 0.05 rejects normality at 5%; p >= 0.05 does not prove normality.')
    if selected_charts.get('summary'):
        addfig(summary_figure(x,r,str(variable)))
    if any(selected_charts.get(k) for k in ('run','hist','box')):
        fig=selected_eda_charts(x,all_x,str(variable),selected_charts.get('run'),
                                selected_charts.get('hist'),selected_charts.get('box'))
        if fig is not None: addfig(fig)
    if strat_series is not None:
        plotdf=pd.DataFrame({'value':all_x,'group':strat_series})
        plotdf=plotdf.loc[np.isfinite(plotdf['value']) & plotdf['group'].notna()]
        strata=list(plotdf.groupby('group',sort=False))
        if 2<=len(strata)<=35:
            fig,ax=plt.subplots(figsize=(max(8,len(strata)*.6),4.5))
            ax.boxplot([g['value'].to_numpy() for _,g in strata],
                       tick_labels=[str(k) for k,_ in strata]);ax.tick_params(axis='x',rotation=45)
            ax.set_title('Stratified box plot');fig.tight_layout();addfig(fig)
    heading('2. Distribution Fit')
    para('AIC and BIC compare candidate models on the same dataset. Lower is better; neither is an absolute goodness-of-fit test. BIC penalizes complexity more strongly. No calibrated GOF p-values are implemented.')
    rows=[['Distribution','AIC','BIC']]
    for _,row in fit_table.iterrows():
        rows.append([row['Distribuzione'],f'{row["AIC"]:.2f}',f'{row["BIC"]:.2f}'])
    table(rows)
    if selected_model in models:
        model=models[selected_model]
        para(f'Selected model: {selected_model}. Parameters: {model["parameters"]}')
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.hist(x,bins='auto',density=True,alpha=.55)
        grid=np.linspace(np.min(x),np.max(x),600)
        ax.plot(grid,model['pdf'](grid),linewidth=2)
        ax.set_title(f'Distribution fit: {selected_model}');fig.tight_layout();addfig(fig)
    heading('3. Capability Analysis')
    if not use_lsl and not use_usl:
        para('No specification limits provided; capability not available.')
    elif use_lsl and use_usl and lsl>=usl:
        para('Invalid specification limits (LSL >= USL).')
    elif selected_model in models:
        model=models[selected_model]
        para(f'LSL: {lsl if use_lsl else "not specified"}; USL: {usl if use_usl else "not specified"}.')
        def fmt(v): return 'N/A' if v is None or not np.isfinite(v) else f'{v:.3f}'
        if selected_model=='Normal':
            mean=float(r['mean']); so=float(r['sd'])
            def idx(sd):
                sides=[]
                if use_lsl:sides.append((mean-lsl)/(3*sd))
                if use_usl:sides.append((usl-mean)/(3*sd))
                return min(sides) if sides else None
            sw=sigma_within if np.isfinite(sigma_within) and sigma_within>0 else None
            table([['Normal capability','Value'],['Cp',fmt((usl-lsl)/(6*sw) if sw and use_lsl and use_usl else None)],
                   ['Cpk',fmt(idx(sw) if sw else None)],['Pp',fmt((usl-lsl)/(6*so) if use_lsl and use_usl else None)],
                   ['Ppk',fmt(idx(so))]])
            para(f'Within estimator: {within_method}. Normal assumption required for classical indices.')
        else:
            try:
                ql=float(model['ppf'](.00135));qm=float(model['ppf'](.5));qu=float(model['ppf'](.99865))
                pstar=(usl-lsl)/(qu-ql) if use_lsl and use_usl else None
                sides=[]
                if use_lsl:sides.append((qm-lsl)/(qm-ql))
                if use_usl:sides.append((usl-qm)/(qu-qm))
                table([['Non-normal overall capability','Value'],['Pp*',fmt(pstar)],['Ppk*',fmt(min(sides))],
                       ['Q0.135%',f'{ql:.5g}'],['Median',f'{qm:.5g}'],['Q99.865%',f'{qu:.5g}']])
                para('Percentile-based overall indices only. No non-normal within index is inferred from the overall fit.')
            except (ValueError,RuntimeError,OverflowError) as e:
                para(f'Non-normal percentiles unavailable: {e}')
        lower=float(model['cdf'](lsl)) if use_lsl else 0.
        upper=max(0.,1.-float(model['cdf'](usl))) if use_usl else 0.
        observed=np.mean(((x<lsl) if use_lsl else np.zeros(len(x),bool)) |
                         ((x>usl) if use_usl else np.zeros(len(x),bool)))*1e6
        table([['Defect rate','PPM'],['Observed',f'{observed:.1f}'],['Model expected',f'{(lower+upper)*1e6:.1f}']])
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.hist(x,bins='auto',density=True,alpha=.55)
        grid=np.linspace(min(np.min(x),lsl if use_lsl else np.min(x)),
                         max(np.max(x),usl if use_usl else np.max(x)),500)
        ax.plot(grid,model['pdf'](grid),linewidth=2)
        if use_lsl:ax.axvline(lsl,linestyle='--',label='LSL')
        if use_usl:ax.axvline(usl,linestyle='--',label='USL')
        ax.legend();ax.set_title('Capability - selected model');fig.tight_layout();addfig(fig)
    heading('4. Stability Analysis')
    if sg is not None:
        para('Rational subgroup analysis: subgroup control charts are available in the application. Review all special-cause signals before capability conclusions.')
        # Include subgroup charts when supported by existing routine.
        try:
            labels=list(range(1,len(sg['sizes'])+1))
            result,err=subgroup_charts(sg,labels,str(variable))
            if not err and result:
                fig,n_bad,kind=result
                para(f'Chart type: Xbar-{kind}; points beyond 3-sigma limits: {n_bad}.')
                addfig(fig)
            elif err:para(err)
        except (ValueError,KeyError,TypeError) as e:para(f'Subgroup chart unavailable: {e}')
    else:
        valid=np.isfinite(all_x)
        pairmask=valid[:-1]&valid[1:]
        mr=np.abs(np.diff(all_x)[pairmask])
        if len(mr):
            mean=float(np.mean(x)); mrbar=float(np.mean(mr)); sigma=mrbar/1.128
            ni=int(np.sum((x>mean+3*sigma)|(x<mean-3*sigma)))
            nm=int(np.sum(mr>3.267*mrbar))
            para(f'I-MR: sigma within = {sigma:.5g}; Individuals outside 3-sigma limits = {ni}; MR outside UCL = {nm}.')
            fig,(a,b)=plt.subplots(2,1,figsize=(9,6))
            ix=np.flatnonzero(valid)+1
            a.plot(ix,x,'o-',markersize=2);a.axhline(mean,color='green')
            a.axhline(mean+3*sigma,color='red',linestyle='--');a.axhline(mean-3*sigma,color='red',linestyle='--')
            a.set_title('Individuals chart')
            b.plot(np.flatnonzero(pairmask)+2,mr,'o-',markersize=2)
            b.axhline(mrbar,color='green');b.axhline(3.267*mrbar,color='red',linestyle='--')
            b.set_title('Moving range chart');fig.tight_layout();addfig(fig)
        else:para('Insufficient consecutive observations for I-MR charts.')
    heading('Interpretation and limitations')
    para('The fitted distribution is a modeling assumption. AIC/BIC selection alone cannot validate the tails or prove process stability. Expected PPM is extrapolated from the fitted model. Check rational subgroup formation, data order, independence, measurement system, and representativeness.')
    doc.build(story)
    out.seek(0)
    return out.getvalue()

def selected_eda_charts(x, all_x, variable, show_run, show_hist, show_box):
    selected = int(show_run) + int(show_hist) + int(show_box)
    if not selected: return None
    fig, axes = plt.subplots(selected, 1, figsize=(11, 3.3*selected), squeeze=False)
    axes = axes.ravel(); i = 0
    if show_run:
        ax = axes[i]; i += 1
        idx = np.flatnonzero(np.isfinite(all_x)) + 1
        ax.plot(idx, x, 'o-', color='#287ca5', markersize=3)
        ax.axhline(np.median(x), color='#b72e34', ls='--', label='Mediana')
        ax.set(title='Run chart (ordine originale delle righe)', xlabel='Riga nel dataset', ylabel=variable)
        ax.legend()
    if show_hist:
        ax = axes[i]; i += 1
        ax.hist(x, bins='auto', color='#8bc7e8', edgecolor='white')
        ax.set(title='Istogramma totale', xlabel=variable, ylabel='Frequenza')
    if show_box:
        ax = axes[i]; i += 1
        ax.boxplot(x, vert=False, patch_artist=True, boxprops={'facecolor':'#8bc7e8'})
        ax.set(title='Box plot totale', xlabel=variable)
    fig.tight_layout()
    return fig

st.subheader("Dataset")
source = st.radio("Origine dati", ["Esempio Thickness", "Carica un file"], horizontal=True)
df = None
if source == "Esempio Thickness":
    try:
        df = demo_data()
    except FileNotFoundError:
        st.error("Manca data/thickness.csv nel repository GitHub.")
else:
    uploaded = st.file_uploader("Excel, CSV o TXT", type=["xlsx", "xls", "csv", "txt"])
    if uploaded is not None:
        try:
            sheet = None
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                sheet = st.selectbox("Foglio Excel", pd.ExcelFile(io.BytesIO(uploaded.getvalue())).sheet_names)
            df = read_uploaded(uploaded.getvalue(), uploaded.name, sheet)
        except Exception as exc:
            st.error(f"Impossibile leggere il file: {exc}")
if df is None:
    st.info("Carica un dataset per iniziare."); st.stop()
if df.empty:
    st.error("Dataset vuoto."); st.stop()
with st.expander(f"Anteprima dati ({len(df)} righe, {len(df.columns)} colonne)"):
    st.dataframe(df.head(20), use_container_width=True)
cols = [c for c in df.columns if np.isfinite(numeric_values(df[c])).sum() >= 8]
if not cols:
    st.warning("Non ci sono colonne con almeno 8 valori numerici."); st.stop()
variable = st.selectbox("Variabile numerica da analizzare", cols, format_func=str)
all_x = numeric_values(df[variable]); x = all_x[np.isfinite(all_x)]
missing = len(all_x) - len(x)
try:
    r = summary(x)
except ValueError as exc:
    st.warning(str(exc)); st.stop()
st.subheader("Sottogruppi razionali (opzionale)")
st.caption("I sottogruppi devono rappresentare misure raccolte in condizioni omogenee e in sequenza temporale. "
           "Non confondere una variabile di stratificazione (es. macchina o turno) con un sottogruppo razionale.")
mode = st.radio("Come sono organizzati i sottogruppi?",
                ["Nessun sottogruppo (dati individuali)", "Colonna identificativa", "Gruppi consecutivi di n osservazioni"],
                horizontal=True)
valid_mask = np.isfinite(all_x)
sg = None
sg_labels = []
sg_groups = []
if mode == "Colonna identificativa":
    options = [c for c in df.columns if c != variable]
    if not options:
        st.warning("Non ci sono altre colonne da usare come identificativo di sottogruppo.")
    else:
        group_col = st.selectbox("Colonna ID sottogruppo", options, format_func=str)
        temp = pd.DataFrame({'value': all_x, 'group': df[group_col].to_numpy()})
        temp = temp.loc[valid_mask & temp['group'].notna()].copy()
        # preserve first-appearance order (not alphabetical)
        for label, frame in temp.groupby('group', sort=False):
            sg_labels.append(str(label))
            sg_groups.append(frame['value'].to_numpy(dtype=float))
        if len(temp) < len(x):
            st.warning(f"Escluse {len(x)-len(temp)} misure valide prive di ID sottogruppo: "
                       "per coerenza saranno escluse anche dalle analisi quando si usano i sottogruppi.")
        if sg_groups and len(temp) >= 8:
            x = temp['value'].to_numpy(dtype=float)
            r = summary(x)
        elif not sg_groups:
            st.warning("Non ci sono sottogruppi validi.")
elif mode == "Gruppi consecutivi di n osservazioni":
    group_n = st.number_input("Osservazioni per sottogruppo", min_value=2, max_value=50, value=5, step=1)
    if missing:
        st.warning("Sono presenti misure mancanti: per non alterare i confini dei sottogruppi, "
                   "i gruppi sono costruiti sulle righe originali. Quelli incompleti non saranno usati.")
    for start in range(0, len(all_x), int(group_n)):
        block = all_x[start:start + int(group_n)]
        if len(block) == int(group_n) and np.all(np.isfinite(block)):
            sg_labels.append(str(start // int(group_n) + 1))
            sg_groups.append(block.astype(float))
    if sg_groups:
        x = np.concatenate(sg_groups)
        if len(x) >= 8:
            r = summary(x)
        else:
            st.warning("Dopo l'esclusione dei gruppi incompleti rimangono meno di 8 osservazioni.")
    st.caption(f"Sottogruppi completi: {len(sg_groups)}; misure incluse: {len(x)}. "
               "L'ultimo gruppo incompleto è escluso.")
if mode != "Nessun sottogruppo (dati individuali)":
    sg, error_sg = subgroup_statistics(sg_groups)
    if error_sg:
        st.warning(error_sg + " La capability within per sottogruppi non sarà calcolata.")
    else:
        st.caption(f"Sottogruppi: {len(sg_groups)} · numerosità: "
                   f"{int(sg['sizes'].min())}–{int(sg['sizes'].max())} · "
                   f"stima σ within: {sg['method']}")

st.caption(f"Osservazioni valide: {len(x)} · Mancanti/non numeriche: {missing} · I valori mancanti sono esclusi dai calcoli")
safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(variable))[:50]

eda, fit, capability, stability = st.tabs(["1 · Exploratory Data Analysis", "2 · Distribution Fit", "3 · Capability Analysis", "4 · Stability Analysis"])

with eda:
    st.subheader("Exploratory Data Analysis")
    st.write("Summary Report, statistiche descrittive e analisi preliminare della distribuzione.")
    st.markdown('**Seleziona i grafici da visualizzare**')
    opt1, opt2, opt3, opt4 = st.columns(4)
    with opt1: show_summary = st.checkbox('Executive Summary', value=True)
    with opt2: show_run = st.checkbox('Run chart', value=True)
    with opt3: show_hist = st.checkbox('Istogramma totale', value=True)
    with opt4: show_box = st.checkbox('Box plot totale', value=True)
    if show_summary:
        show_png(summary_figure(x, r, str(variable)), f'EDEAS_{safe_name}_Summary.png')
    chart = selected_eda_charts(all_x[np.isfinite(all_x)], all_x, str(variable), show_run, show_hist, show_box)
    if chart is not None:
        show_png(chart, f'EDEAS_{safe_name}_EDA_Charts.png')
    if show_run:
        st.caption('Il run chart segue l’ordine originale delle righe, non un asse temporale verificato. I valori mancanti creano intervalli nell’indice.')
    q1, q3 = r["q"][1], r["q"][3]
    outliers = int(np.sum((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))))
    st.markdown("**Commento statistico guidato**")
    if r["p"] < .05:
        st.warning(f"Anderson–Darling: p = {r['p']:.4g}. Al livello del 5% si rifiuta l'ipotesi di normalità. Valuta Distribution Fit e possibili sottogruppi.")
    else:
        st.info(f"Anderson–Darling: p = {r['p']:.4g}. Non si rifiuta la normalità al 5%; questo non prova che i dati siano normali.")
    st.write(f"Media = {r['mean']:.3f}; deviazione standard = {r['sd']:.3f}; possibili outlier secondo Tukey = {outliers}.")
    st.markdown("**Box plot stratificato**")
    strat_options = [c for c in df.columns if c != variable]
    if strat_options:
        strat_col = st.selectbox("Stratifica per (macchina, turno, lotto, operatore, ecc.)", ["—"] + strat_options, format_func=str)
        if strat_col != "—":
            plotdf = pd.DataFrame({'value': all_x, 'group': df[strat_col].to_numpy()})
            plotdf = plotdf.loc[np.isfinite(plotdf['value']) & plotdf['group'].notna()]
            strata = list(plotdf.groupby('group', sort=False))
            if 2 <= len(strata) <= 35:
                fig, ax = plt.subplots(figsize=(max(9, len(strata) * .6), 5))
                ax.boxplot([g['value'].to_numpy() for _, g in strata], tick_labels=[str(k) for k, _ in strata],
                           patch_artist=True, boxprops={'facecolor': '#8bc7e8'})
                ax.set_title(f"Box plot stratificato — {variable} per {strat_col}")
                ax.set_ylabel(str(variable)); ax.tick_params(axis='x', rotation=45)
                fig.tight_layout()
                show_png(fig, f"EDEAS_{safe_name}_Stratified_Boxplot.png")
                st.dataframe(pd.DataFrame([{'Strato': str(k), 'N': len(g), 'Media': g['value'].mean(),
                                            'StDev': g['value'].std(ddof=1)} for k, g in strata]),
                             hide_index=True, use_container_width=True)
            else:
                st.warning("Per il box plot servono da 2 a 35 categorie non vuote.")
    else:
        st.caption("Per stratificare, carica un dataset con almeno una colonna categoriale aggiuntiva.")
    with st.expander("Domande guidate — EDA"):
        question = st.selectbox("Cosa vuoi approfondire?", ["Cosa significa il p-value?", "Come interpreto gli outlier?", "Quale analisi eseguire dopo?"])
        if question.startswith("Cosa significa"):
            st.write("Il p-value misura quanto il risultato del test sia incompatibile con l'ipotesi di normalità. Non rappresenta la probabilità che i dati siano normali.")
        elif "outlier" in question:
            st.write("La regola 1,5 × IQR segnala valori insoliti, ma non dimostra che siano errori. Verifica cause, acquisizione dati e sottogruppi prima di eliminarli.")
        else:
            st.write("Controlla l'ordine temporale in Stability Analysis; valuta il modello probabilistico in Distribution Fit; usa Capability Analysis quando disponi dei limiti di specifica.")
    st.caption("IC della deviazione standard basato sull'ipotesi di normalità; IC della mediana non parametrico. Il p-value AD può differire da Minitab.")

with fit:
    st.subheader('Distribution Fit')
    st.write('Confronto Normal, Lognormal, Weibull, Gamma e miscele gaussiane (GMM).')
    use_gmm = st.checkbox('Includi modelli bimodali e multimodali (Gaussian Mixture)', value=True)
    max_gmm = st.slider('Numero massimo di componenti GMM', 2, 5, 3, disabled=not use_gmm)
    with st.expander('Come viene scelta la distribuzione? AIC, BIC e goodness-of-fit', expanded=True):
        st.markdown('**AIC = 2k - 2 log(L)** e **BIC = k log(n) - 2 log(L)**, dove k è il numero di parametri stimati, n il numero di osservazioni e L la verosimiglianza massimizzata.')
        st.write('Entrambi si minimizzano. AIC privilegia l’accuratezza predittiva relativa; BIC penalizza più fortemente la complessità e aiuta a evitare miscele con troppe componenti.')
        st.write('**Delta BIC**: differenza rispetto al BIC minimo. Valori piccoli indicano modelli competitivi; differenze elevate favoriscono il modello migliore secondo BIC.')
        st.warning('**Goodness-of-fit non ancora implementato per ogni distribuzione.** AIC e BIC non sono p-value. Il p-value Anderson–Darling presente nell’EDA testa soltanto la normalità; non va usato per accettare automaticamente Lognormal, Weibull, Gamma o GMM.')
        st.caption('Per confrontare le code e stimare PPM molto bassi sono necessari controlli diagnostici e, idealmente, test calibrati mediante bootstrap parametrico.')
    if len(x) < 20 and use_gmm:
        st.warning('Per stimare le miscele sono richieste almeno 20 osservazioni; con campioni piccoli la stima è fragile.')
    models, fit_table = fit_models(x, max_gmm, use_gmm)
    if fit_table.empty:
        st.warning('Non è stato possibile stimare un modello.')
    else:
        fit_table = fit_table.copy()
        fit_table['Delta AIC'] = fit_table['AIC'] - fit_table['AIC'].min()
        fit_table['Delta BIC'] = fit_table['BIC'] - fit_table['BIC'].min()
        st.dataframe(fit_table, hide_index=True, use_container_width=True)
        st.caption('La raccomandazione automatica usa il BIC minimo; la selezione manuale prevale. Un buon punteggio relativo non garantisce un fit adeguato.')
        recommended = str(fit_table.iloc[0]['Distribuzione'])
        selected = st.selectbox('Modello da utilizzare anche nella Capability Overall',
                                list(models), index=list(models).index(recommended), key='chosen_fit')
        model = models[selected]
        st.session_state['fit_selection'] = selected
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.hist(x, bins='auto', density=True, color='#8bc7e8', edgecolor='white', alpha=.8)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 700)
        ax.plot(grid, model['pdf'](grid), color='#b72e34', lw=2, label=selected)
        ax.set(title=f'Distribution Fit — {variable}', xlabel=str(variable), ylabel='Densità')
        ax.legend()
        show_png(fig, f'EDEAS_{safe_name}_Distribution_Fit.png')
        # Probability plot: model quantiles against empirical ordered values
        try:
            prob = (np.arange(1, len(x)+1) - .5)/len(x)
            theory = np.array([model['ppf'](float(q)) for q in prob])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(theory, np.sort(x), s=14, alpha=.7)
            lo = min(float(np.min(theory)), float(np.min(x)))
            hi = max(float(np.max(theory)), float(np.max(x)))
            ax.plot([lo, hi], [lo, hi], '--', color='#b72e34')
            ax.set(title=f'Probability plot — {selected}', xlabel='Quantili teorici', ylabel='Quantili osservati')
            fig.tight_layout()
            show_png(fig, f'EDEAS_{safe_name}_Probability_Plot.png')
        except (ValueError, RuntimeError):
            st.warning('Probability plot non disponibile per questo modello.')
        st.info(f'Modello con BIC minimo: {recommended}. Selezionato: {selected}. Controlla soprattutto le code e le cause fisiche di eventuale multimodalità.')
        st.caption('Il fit e il confronto AIC/BIC non equivalgono a un test goodness-of-fit. P-value calibrati per i modelli con parametri stimati non sono ancora implementati.')

with capability:
    st.subheader("Capability Analysis")
    st.write("Analisi preliminare per dati individuali con limiti di specifica. La stabilità va verificata separatamente.")
    c1, c2 = st.columns(2)
    with c1:
        use_lsl = st.checkbox("Imposta LSL", value=True)
        lsl = st.number_input("Lower Specification Limit (LSL)", value=float(np.min(x)), disabled=not use_lsl, format="%.4f")
    with c2:
        use_usl = st.checkbox("Imposta USL", value=True)
        usl = st.number_input("Upper Specification Limit (USL)", value=float(np.max(x)), disabled=not use_usl, format="%.4f")
    if not use_lsl and not use_usl:
        st.warning("Inserisci almeno un limite di specifica.")
    elif use_lsl and use_usl and lsl >= usl:
        st.error("LSL deve essere inferiore a USL.")
    else:
        # Fit calcolato anche quando l’utente apre direttamente il tab Capability.
        cap_models, cap_table = fit_models(x, 3, True)
        if cap_table.empty:
            st.error('Nessun modello di distribuzione stimabile.'); st.stop()
        default_model = st.session_state.get('fit_selection', str(cap_table.iloc[0]['Distribuzione']))
        if default_model not in cap_models:
            default_model = str(cap_table.iloc[0]['Distribuzione'])
        chosen = st.selectbox('Modello probabilistico per capability Overall e PPM attesi',
                              list(cap_models), index=list(cap_models).index(default_model), key='cap_model')
        cap_model = cap_models[chosen]
        st.caption('La Capability usa il modello scelto qui (preselezionato dal Distribution Fit, se disponibile). La Within resta basata su MR/sottogruppi con ipotesi normale.')
        sigma_overall = float(r["sd"])
        if mode == "Nessun sottogruppo (dati individuali)":
            if missing:
                st.warning("Le misure mancanti interrompono la continuità temporale: la stima within da MR "
                           "viene calcolata solo sulle coppie di righe originali entrambe valide.")
            pairs = np.isfinite(all_x[1:]) & np.isfinite(all_x[:-1])
            moving_ranges = np.abs(np.diff(all_x)[pairs])
            sigma_within = float(moving_ranges.mean() / 1.128) if len(moving_ranges) else float('nan')
            within_method = "MR̄/d₂ (moving range di 2, solo coppie temporalmente adiacenti)"
        else:
            sigma_within = sg['sigma'] if sg is not None else float('nan')
            within_method = sg['method'] if sg is not None else "Non stimabile"
        if not np.isfinite(sigma_within) or sigma_within <= 0:
            st.warning("La deviazione standard within non è stimabile o è nulla: Cp e Cpk non disponibili.")
        else:
            st.caption(f"σ within = {sigma_within:.5g} · metodo: {within_method}")
        def one_sided(sd):
            lower = (r["mean"] - lsl) / (3 * sd) if use_lsl else float("inf")
            upper = (usl - r["mean"]) / (3 * sd) if use_usl else float("inf")
            return min(lower, upper)
        cp = (usl - lsl) / (6 * sigma_within) if use_lsl and use_usl and np.isfinite(sigma_within) and sigma_within > 0 else None
        pp = (usl - lsl) / (6 * sigma_overall) if use_lsl and use_usl else None
        cpk = one_sided(sigma_within) if np.isfinite(sigma_within) and sigma_within > 0 else None
        ppk = one_sided(sigma_overall)
        if chosen == 'Normal':
            m1, m2, m3, m4 = st.columns(4)
            for col, label, val in [(m1, "Cp", cp), (m2, "Cpk", cpk), (m3, "Pp", pp), (m4, "Ppk", ppk)]:
                col.metric(label, f"{val:.3f}" if val is not None else "N/D")
        else:
            st.info('Modello non normale: Cp/Cpk/Pp/Ppk normali sono nascosti. Mostriamo soltanto la capability Overall basata sui percentili. La Within non normale richiede un metodo specifico.')
            with st.expander('Confronto didattico: indici sotto ipotesi normale (non usare per questo modello)'):
                m1, m2, m3, m4 = st.columns(4)
                for col, label, val in [(m1, "Cp normale", cp), (m2, "Cpk normale", cpk), (m3, "Pp normale", pp), (m4, "Ppk normale", ppk)]:
                    col.metric(label, f"{val:.3f}" if val is not None else "N/D")
        # Percentile-based non-normal overall indices, based on the selected fitted model.
        try:
            qlo = float(cap_model['ppf'](.00135))
            qmed = float(cap_model['ppf'](.5))
            qhi = float(cap_model['ppf'](.99865))
            valid_quantiles = np.isfinite([qlo,qmed,qhi]).all() and qlo < qmed < qhi
        except (ValueError, RuntimeError, OverflowError):
            valid_quantiles = False
        if valid_quantiles and chosen != 'Normal':
            pp_star = (usl-lsl)/(qhi-qlo) if use_lsl and use_usl else None
            sides = []
            if use_lsl: sides.append((qmed-lsl)/(qmed-qlo))
            if use_usl: sides.append((usl-qmed)/(qhi-qmed))
            ppk_star = min(sides)
            a,b = st.columns(2)
            a.metric('Pp* — percentili 99,73%', f'{pp_star:.3f}' if pp_star is not None else 'N/D')
            b.metric('Ppk* — percentili / mediana', f'{ppk_star:.3f}')
            st.caption(f'Modello {chosen}: Q0,135% = {qlo:.5g}; Q50% = {qmed:.5g}; Q99,865% = {qhi:.5g}. Indici Overall secondo convenzione percentile-based.')
        elif chosen != 'Normal':
            st.warning('Impossibile calcolare i quantili del modello selezionato.')
        observed = np.mean(((x < lsl) if use_lsl else np.zeros(len(x), dtype=bool)) |
                           ((x > usl) if use_usl else np.zeros(len(x), dtype=bool))) * 1e6
        st.metric("PPM osservati fuori specifica", f"{observed:,.0f}")
        def expected_ppm(sd):
            lower = stats.norm.cdf(lsl, loc=r['mean'], scale=sd) if use_lsl else 0
            upper = stats.norm.sf(usl, loc=r['mean'], scale=sd) if use_usl else 0
            return 1e6 * (lower + upper)
        try:
            lower_prob = float(cap_model['cdf'](lsl)) if use_lsl else 0.0
            upper_prob = 1.0 - float(cap_model['cdf'](usl)) if use_usl else 0.0
            st.metric(f'PPM attesi — modello {chosen} (Overall)', f'{1e6*(lower_prob+upper_prob):,.1f}')
        except (ValueError, RuntimeError):
            st.warning('PPM del modello selezionato non calcolabili.')
        if chosen == 'Normal':
            m_within, m_overall = st.columns(2)
            with m_within:
                st.metric("PPM attesi — Within (normale)",
                          f"{expected_ppm(sigma_within):,.1f}" if np.isfinite(sigma_within) and sigma_within > 0 else "N/D")
            with m_overall:
                st.metric("PPM attesi — Overall (normale)", f"{expected_ppm(sigma_overall):,.1f}")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.hist(x, bins="auto", density=True, color="#8bc7e8", edgecolor="white", alpha=.8)
        grid = np.linspace(min(np.min(x), lsl if use_lsl else np.min(x)), max(np.max(x), usl if use_usl else np.max(x)), 500)
        ax.plot(grid, cap_model["pdf"](grid), color="#b72e34", lw=2, label=f"{chosen} (overall)")
        if chosen == 'Normal' and np.isfinite(sigma_within) and sigma_within > 0:
            ax.plot(grid, stats.norm.pdf(grid, r["mean"], sigma_within), color="#287ca5", lw=2,
                    linestyle="--", label="Normal (within)")
        if use_lsl: ax.axvline(lsl, color="darkorange", ls="--", label="LSL")
        if use_usl: ax.axvline(usl, color="purple", ls="--", label="USL")
        ax.set_title("Capability overview — individual data"); ax.set_xlabel(str(variable)); ax.set_ylabel("Density"); ax.legend()
        show_png(fig, f"EDEAS_{safe_name}_Capability.png")
        if r["p"] < .05:
            st.warning("Commento guidato: il test AD rifiuta la normalità. Gli indici normali sono mostrati a scopo didattico e potrebbero essere fuorvianti; serve una capability non normale validata.")
        else:
            st.info("Commento guidato: la normalità non è stata rifiutata. Prima di interpretare Cp/Cpk verifica stabilità e adeguatezza del sistema di misura.")
        st.caption("Sigma overall = deviazione standard campionaria. Cp/Cpk utilizzano sigma within; Pp/Ppk utilizzano sigma overall. Cp e Pp richiedono entrambi i limiti. PPM osservati: proporzione campionaria. PPM attesi: stime modellistiche sotto ipotesi normale, non garanzie. Inserisci le vere specifiche. I risultati presuppongono dati rappresentativi e un modello appropriato.")

with stability:
    st.subheader("Stability Analysis")
    if mode != "Nessun sottogruppo (dati individuali)":
        st.write("Carte X̄-R per sottogruppi di 2–10 osservazioni; X̄-S per sottogruppi più grandi. "
                 "I sottogruppi sono rappresentati nell'ordine in cui compaiono nel file.")
        if sg is None:
            st.warning("Impossibile costruire le carte: controlla la struttura dei sottogruppi.")
        else:
            result, error = subgroup_charts(sg, sg_labels, str(variable))
            if error:
                st.warning(error)
            else:
                fig, n_bad, kind = result
                show_png(fig, f"EDEAS_{safe_name}_Xbar_{kind}.png")
                st.write(f"**Segnali oltre 3σ:** {n_bad} punti complessivi sulle due carte.")
                if n_bad:
                    st.warning("Sono presenti punti oltre i limiti di controllo. Indaga cause speciali "
                               "prima di interpretare la capability.")
                else:
                    st.info("Nessun punto oltre 3σ. Non è sufficiente per concludere che il processo "
                            "sia stabile: sono necessari anche altri test di sequenza e trend.")
            st.caption("Limiti Phase I stimati dai dati; assunti sottogruppi razionali indipendenti, "
                       "ordinati temporalmente e di uguale dimensione per le carte X̄-R/X̄-S.")
    else:
        st.write("Carte I-MR per osservazioni individuali, nell'ordine in cui compaiono nel file.")
        st.warning("L'ordine delle righe deve essere cronologico. Se il file non è ordinato nel tempo, riordinalo prima dell'analisi. Non eliminare i valori mancanti senza valutare l'effetto sui moving range.")
        if missing:
            st.warning(f"Sono stati esclusi {missing} valori: moving range calcolati solo tra righe originali adiacenti entrambe valide.")
        pairmask = np.isfinite(all_x[1:]) & np.isfinite(all_x[:-1])
        mr = np.abs(np.diff(all_x)[pairmask])
        mrbar = np.mean(mr) if len(mr) else float('nan')
        sigma = mrbar / 1.128
        mean = np.mean(x)
        ucl_i, lcl_i = mean + 3 * sigma, mean - 3 * sigma
        ucl_mr = 3.267 * mrbar
        bad_i = np.flatnonzero((x > ucl_i) | (x < lcl_i))
        bad_mr = np.flatnonzero(mr > ucl_mr)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
        idx = np.flatnonzero(np.isfinite(all_x)) + 1
        ax1.plot(idx, x, "o-", color="#287ca5", markersize=3)
        ax1.axhline(mean, color="green", label="CL")
        ax1.axhline(ucl_i, color="#b72e34", ls="--", label="UCL / LCL")
        ax1.axhline(lcl_i, color="#b72e34", ls="--")
        if len(bad_i): ax1.scatter(idx[bad_i], x[bad_i], color="red", s=65, zorder=5)
        ax1.set_title("Individuals Chart"); ax1.set_ylabel(str(variable)); ax1.legend(loc="best")
        midx = (np.flatnonzero(pairmask) + 2)
        ax2.plot(midx, mr, "o-", color="#287ca5", markersize=3)
        ax2.axhline(mrbar, color="green", label="CL")
        ax2.axhline(ucl_mr, color="#b72e34", ls="--", label="UCL")
        ax2.axhline(0, color="#b72e34", ls="--", label="LCL = 0")
        if len(bad_mr): ax2.scatter(midx[bad_mr], mr[bad_mr], color="red", s=65, zorder=5)
        ax2.set_title("Moving Range Chart (MR of 2)"); ax2.set_xlabel("Observation order"); ax2.set_ylabel("Moving range")
        ax2.legend(loc="best")
        fig.tight_layout()
        show_png(fig, f"EDEAS_{safe_name}_I_MR.png")
        st.write(f"**Punti oltre i limiti 3σ:** Individuals = {len(bad_i)}; Moving Range = {len(bad_mr)}.")
        if len(bad_i) or len(bad_mr):
            st.warning("Commento guidato: sono presenti segnali oltre i limiti di controllo. Indaga possibili cause speciali prima di valutare la capability.")
        else:
            st.info("Commento guidato: nessun punto oltre i limiti 3σ. Questo non dimostra da solo la stabilità: servono anche test per run, trend e altre cause speciali.")
        st.caption("Limiti Phase I stimati dagli stessi dati. Regola implementata: punti oltre 3σ; test aggiuntivi non ancora implementati. Le carte I-MR presuppongono osservazioni individuali temporalmente ordinate.")

st.divider()
st.caption("EDEAS Training Edition · Analisi deterministiche Python · Commenti guidati gratuiti (non AI generativa) · I risultati vanno interpretati nel contesto del processo.")


st.divider()
st.subheader('5 · Report PDF completo')
st.write('Scarica un documento con le quattro sezioni, le tabelle, il modello selezionato, i grafici EDA scelti e la capability coerente con il modello.')
if 'fit_table' not in locals() or fit_table.empty:
    st.warning('Prima occorre stimare almeno una distribuzione.')
else:
    pdf_model = chosen if 'chosen' in locals() else selected
    pdf_models = cap_models if 'cap_models' in locals() else models
    pdf_lsl = lsl if 'lsl' in locals() else float(np.min(x))
    pdf_usl = usl if 'usl' in locals() else float(np.max(x))
    pdf_use_lsl = use_lsl if 'use_lsl' in locals() else False
    pdf_use_usl = use_usl if 'use_usl' in locals() else False
    pdf_sigma = sigma_within if 'sigma_within' in locals() else float('nan')
    pdf_method = within_method if 'within_method' in locals() else 'not estimated'
    pdf_strat = df[strat_col].to_numpy() if 'strat_col' in locals() and strat_col != '—' else None
    if st.button('Prepara report PDF', type='primary'):
        with st.spinner('Creazione del report in corso...'):
            try:
                st.session_state['pdf_report_bytes'] = make_pdf_report(
                    x, all_x, r, str(variable), fit_table, pdf_models, pdf_model,
                    pdf_lsl, pdf_usl, pdf_use_lsl, pdf_use_usl, pdf_sigma, pdf_method,
                    sg, {'summary':show_summary,'run':show_run,'hist':show_hist,'box':show_box},pdf_strat)
                st.session_state['pdf_report_signature'] = (str(variable),len(x),float(np.mean(x)),float(np.std(x)),pdf_model,pdf_lsl,pdf_usl,pdf_use_lsl,pdf_use_usl,
                    show_summary,show_run,show_hist,show_box,strat_col if 'strat_col' in locals() else None)
            except Exception as exc:
                st.error(f'Impossibile creare il PDF: {exc}')
    signature = (str(variable),len(x),float(np.mean(x)),float(np.std(x)),pdf_model,pdf_lsl,pdf_usl,pdf_use_lsl,pdf_use_usl,
                 show_summary,show_run,show_hist,show_box,strat_col if 'strat_col' in locals() else None)
    if st.session_state.get('pdf_report_signature') == signature and st.session_state.get('pdf_report_bytes'):
        st.download_button('Scarica report PDF',data=st.session_state['pdf_report_bytes'],
            file_name=f'EDEAS_{safe_name}_Full_Analysis.pdf',mime='application/pdf')
    else:
        st.caption('Premi Prepara report PDF dopo aver completato le scelte. Se cambi le impostazioni, rigenera il report.')
