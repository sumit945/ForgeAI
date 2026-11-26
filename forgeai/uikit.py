"""Design system for the ForgeAI console - "lavender glass" theme (soft gradient background, white glass cards, purple gradient actions,
pill navigation, circular score rings), modelled on the reference mock-ups.

Tokens are CSS variables (light theme + a dark palette), components are small functions returning HTML strings.
Every dynamic string is escaped here, so pages can pass raw run data safely.
"""
from __future__ import annotations

from html import escape as esc
from urllib.parse import quote

LIGHT = {
    "text": "#1E1B4B", "text-2": "#52506B", "muted": "#8584A4", "inverse": "#FFFFFF",
    "surface": "rgba(255,255,255,.92)", "surface-solid": "#FFFFFF", "subtle": "#F3F1FF", "border": "rgba(124,108,247,.16)", "border-strong": "#DAD6FA",
    "brand": "#6C5CE7", "brand-2": "#5B4BEA", "brand-grad": "linear-gradient(135deg,#8B7CF8 0%,#5B4BEA 100%)", "brand-subtle": "#ECE9FF", "brand-on": "#4C3FD1",
    "ok": "#22C55E", "ok-bg": "#E5F8EC", "ok-fg": "#15803D", "warn": "#F59E0B", "warn-bg": "#FEF3D7", "warn-fg": "#9A5B05",
    "bad": "#EF4444", "bad-bg": "#FDE7E7", "bad-fg": "#B42323", "info": "#3B82F6", "info-bg": "#E3EEFF", "info-fg": "#1D4ED8",
    "ring-track": "#E7E4FA", "sidebar": "rgba(255,255,255,.78)", "shadow": "0 10px 30px rgba(91,75,234,.08)", "shadow-sm": "0 2px 8px rgba(91,75,234,.07)",
    "bg": "radial-gradient(1100px 560px at 6% -8%,rgba(139,124,248,.24),transparent 60%),radial-gradient(900px 520px at 100% 0%,rgba(96,165,250,.20),transparent 55%),linear-gradient(135deg,#F1F0FF 0%,#F8F7FF 48%,#EDF3FF 100%)",
}
DARK = {
    "text": "#ECEBFF", "text-2": "#B9B7DA", "muted": "#8E8CB5", "inverse": "#12112B",
    "surface": "rgba(30,28,66,.80)", "surface-solid": "#1B1A3F", "subtle": "#24224F", "border": "rgba(139,124,248,.22)", "border-strong": "#3A3779",
    "brand": "#8B7CF8", "brand-2": "#7C6CF7", "brand-grad": "linear-gradient(135deg,#9A8DFF 0%,#6E5DF0 100%)", "brand-subtle": "#2B2862", "brand-on": "#CFC9FF",
    "ok": "#34D399", "ok-bg": "#12301F", "ok-fg": "#86EFAC", "warn": "#FBBF24", "warn-bg": "#3A2A0A", "warn-fg": "#FCD34D",
    "bad": "#F87171", "bad-bg": "#3A1517", "bad-fg": "#FCA5A5", "info": "#60A5FA", "info-bg": "#0F2A4D", "info-fg": "#93C5FD",
    "ring-track": "#2F2C66", "sidebar": "rgba(22,21,52,.85)", "shadow": "0 10px 30px rgba(0,0,0,.35)", "shadow-sm": "0 2px 8px rgba(0,0,0,.3)",
    "bg": "radial-gradient(1100px 560px at 6% -8%,rgba(139,124,248,.18),transparent 60%),linear-gradient(135deg,#0F0E26 0%,#15143A 55%,#0E1230 100%)",
}

ICONS = {
    "dashboard": '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    "build": '<rect x="3" y="3" width="18" height="18" rx="4"/><path d="M12 8v8M8 12h8"/>',
    "evaluate": '<path d="m9 11 3 3 8-8"/><path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
    "optimize": '<path d="M21 12a9 9 0 0 0-15-6.7L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 15 6.7L21 16"/><path d="M16 16h5v5"/>',
    "runner": '<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 8V5"/><circle cx="12" cy="3.8" r="1.2"/><circle cx="9" cy="14" r="1"/><circle cx="15" cy="14" r="1"/>',
    "projects": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "reports": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
    "settings": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "layers": '<path d="M12 2 2 7l10 5 10-5z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>', "bars": '<path d="M12 20V10M18 20V4M6 20v-4"/>',
    "trophy": '<path d="M8 21h8M12 17v4"/><path d="M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3"/>',
    "check": '<path d="m5 12 5 5 9-10"/>', "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "x": '<path d="M18 6 6 18M6 6l12 12"/>', "chev": '<path d="m9 18 6-6-6-6"/>', "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>', "back": '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>', "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "download": '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>', "bolt": '<path d="M13 3 4 14h7l-1 7 9-11h-7z"/>', "bell": '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/>',
    "ok": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>', "info": '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/>', "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "send": '<path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/>', "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15 15 0 0 1 0 20a15 15 0 0 1 0-20"/>',
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>', "tool": '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.4 2.4-2.6-.6-.6-2.6z"/>',
    "scissors": '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.1 15.9M14.5 14.5 20 20M8.1 8.1 12 12"/>', "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "thumb": '<path d="M7 11v9H4v-9zM7 11l4-8a2.5 2.5 0 0 1 2.5 3L13 10h6a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 18.8 21H7"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", width: float = 1.75) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg" style="flex:none">{ICONS[name]}</svg>')


def _data_uri(name: str, color: str) -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
           f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
    return f'url("data:image/svg+xml;utf8,{quote(svg, safe="/:=;,() ")}")'


PAGES = [("dashboard", "Dashboard"), ("build", "Build Agent"), ("evaluate", "Evaluate"), ("optimize", "Optimize"), ("runner", "Agent Runner"),
         ("projects", "Projects"), ("reports", "Reports"), ("settings", "Settings")]


def stylesheet(theme: str = "light", active: str = "dashboard") -> str:
    t = DARK if theme == "dark" else LIGHT
    root = ";".join(f"--{k}:{v}" for k, v in t.items())
    off, on = ("#A6A4C8", "#FFFFFF") if theme == "dark" else ("#8584A4", "#4C3FD1")
    nav = []
    for key, _ in PAGES:
        is_on = key == active
        nav.append(f".st-key-nav_{key} button{{background:{'var(--brand-subtle)' if is_on else 'transparent'} !important;color:{on if is_on else 'var(--text-2)'} !important;font-weight:{600 if is_on else 500} !important;}}"
                   f".st-key-nav_{key} button::before{{content:'';width:20px;height:20px;margin-right:12px;background:{_data_uri(key, on if is_on else off)} center/contain no-repeat;display:inline-block;flex:none;}}")
    return f"""<style>
:root{{{root}}}
html,body,[data-testid="stApp"],.stApp{{font-family:Inter,"Segoe UI","Nirmala UI","Noto Sans","Noto Sans Devanagari",system-ui,sans-serif !important;background:var(--bg) !important;background-attachment:fixed !important;color:var(--text);}}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stDecoration"]{{display:none !important;}}
.block-container{{padding:72px 36px 56px !important;max-width:1240px !important;position:relative;}}
h1,h2,h3,h4{{color:var(--text) !important;letter-spacing:-.01em;}} p,li,label,span{{color:inherit;}}
[data-testid="stSidebar"]{{background:var(--sidebar) !important;backdrop-filter:blur(16px);width:248px !important;min-width:248px !important;border-right:1px solid var(--border);}}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{{padding:22px 14px 16px;}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{{gap:6px;}}
[data-testid="stSidebar"] .stButton button{{justify-content:flex-start;width:100%;border:0;border-radius:12px;padding:10px 14px;font-size:14px;box-shadow:none;}}
[data-testid="stSidebar"] .stButton button:hover{{background:var(--brand-subtle) !important;}}
[data-testid="stSidebar"] .stButton button p{{margin:0;color:inherit !important;}}
[data-testid="stSidebar"] .stButton button>div{{justify-content:flex-start !important;text-align:left !important;width:100%;}}
{''.join(nav)}
[data-testid="stBaseButton-primary"]{{background:var(--brand-grad) !important;color:#fff !important;border:0 !important;border-radius:12px !important;font-weight:600;padding:.55rem 1.1rem;box-shadow:0 8px 18px rgba(91,75,234,.30);}}
[data-testid="stBaseButton-primary"]:hover{{filter:brightness(1.06);box-shadow:0 10px 22px rgba(91,75,234,.38);}}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"]{{background:var(--surface-solid);color:var(--brand-on) !important;border:1px solid var(--border-strong);border-radius:12px;font-weight:600;}}
[data-testid="stBaseButton-secondary"]:hover{{background:var(--brand-subtle);border-color:var(--brand);}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{{background:transparent;border:0;}}
[data-testid="stVerticalBlockBorderWrapper"]{{background:var(--surface);border:1px solid var(--border) !important;border-radius:18px !important;box-shadow:var(--shadow);padding:6px 10px;}}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-baseweb="select"]>div,[data-testid="stNumberInput"] input{{border-radius:12px !important;border-color:var(--border-strong) !important;background:var(--surface-solid) !important;color:var(--text) !important;font-size:14px;}}
[data-testid="stFileUploader"] section{{background:var(--subtle);border:1.5px dashed var(--border-strong);border-radius:14px;}}
[data-baseweb="tab-list"]{{gap:8px;border-bottom:0;}}
[data-baseweb="tab"]{{font-weight:600;font-size:13px;color:var(--brand-on);padding:7px 16px;background:var(--surface-solid);border:1px solid var(--border-strong);border-radius:10px;height:auto;}}
[data-baseweb="tab"][aria-selected="true"]{{background:var(--brand-subtle);border-color:var(--brand);color:var(--brand-on);}}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{{display:none !important;}}
[data-testid="stChatMessage"]{{background:transparent;border:0;padding:6px 0;}}
[data-testid="stChatInput"]{{border-radius:14px;}}
[data-testid="stExpander"]{{border:1px solid var(--border);border-radius:16px;background:var(--surface);}}
[data-testid="stDataFrame"]{{border:1px solid var(--border);border-radius:12px;}}
.fx-top{{position:absolute;right:36px;top:16px;display:flex;gap:14px;align-items:center;z-index:5}}
.fx-top .ic{{width:36px;height:36px;border-radius:50%;background:var(--surface-solid);box-shadow:var(--shadow-sm);display:flex;align-items:center;justify-content:center;color:var(--text-2)}}
.fx-top .av{{width:36px;height:36px;border-radius:50%;background:var(--brand-grad);color:#fff;font-weight:600;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(91,75,234,.35)}}
.fx-h1{{font-size:24px;font-weight:700;line-height:32px;letter-spacing:-.3px;color:var(--text);margin:0}}
.fx-sub{{font-size:14px;line-height:21px;color:var(--text-2);margin:4px 0 0}}
.fx-h3{{font-size:15px;font-weight:700;line-height:22px;color:var(--text);margin:0}}
.fx-muted{{font-size:12px;line-height:18px;color:var(--muted)}} .fx-caption{{font-size:11px;font-weight:700;letter-spacing:.6px;color:var(--muted);text-transform:uppercase}}
.fx-card{{background:var(--surface);border:1px solid var(--border);border-radius:18px;box-shadow:var(--shadow);padding:20px;backdrop-filter:blur(10px);overflow-x:auto}}
.fx-grid{{display:grid;gap:18px}}.fx-g4{{grid-template-columns:repeat(4,1fr)}}.fx-g3{{grid-template-columns:repeat(3,1fr)}}.fx-g2{{grid-template-columns:repeat(2,1fr)}}.fx-g5{{grid-template-columns:repeat(5,1fr)}}
.fx-hero{{display:flex;justify-content:space-between;align-items:center;gap:24px;overflow:hidden;position:relative;min-height:230px;background:linear-gradient(120deg,rgba(255,255,255,.95) 0%,rgba(236,233,255,.9) 100%)}}
.fx-hero .t{{max-width:520px}} .fx-hero .w{{font-size:32px;font-weight:800;color:var(--text);letter-spacing:-.6px;line-height:40px}}
.fx-hero .g{{font-size:20px;font-weight:700;background:var(--brand-grad);-webkit-background-clip:text;background-clip:text;color:transparent;margin:6px 0 10px}}
.fx-tile{{display:flex;align-items:center;gap:14px;padding:16px 18px}}.fx-tile .ic{{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;flex:none}}
.fx-tile .v{{font-size:26px;font-weight:700;color:var(--text);line-height:30px}}.fx-tile .l{{font-size:12px;color:var(--muted)}}
.fx-kpi{{display:flex;flex-direction:column;gap:6px}}.fx-kpi .l{{font-size:12px;font-weight:600;color:var(--muted)}}.fx-kpi .v{{font-size:30px;font-weight:700;line-height:36px;letter-spacing:-.5px;color:var(--text)}}.fx-kpi .h{{font-size:12px;color:var(--text-2)}}
.fx-badge{{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:2px 10px 2px 8px;font-size:12px;font-weight:600;line-height:18px;white-space:nowrap}}.fx-badge i{{width:6px;height:6px;border-radius:50%;display:inline-block}}
.fx-ok{{background:var(--ok-bg);color:var(--ok-fg)}}.fx-ok i{{background:var(--ok)}}.fx-warn{{background:var(--warn-bg);color:var(--warn-fg)}}.fx-warn i{{background:var(--warn)}}
.fx-bad{{background:var(--bad-bg);color:var(--bad-fg)}}.fx-bad i{{background:var(--bad)}}.fx-info{{background:var(--info-bg);color:var(--info-fg)}}.fx-info i{{background:var(--info)}}
.fx-brand{{background:var(--brand-subtle);color:var(--brand-on)}}.fx-brand i{{background:var(--brand)}}.fx-neutral{{background:var(--subtle);color:var(--text-2)}}.fx-neutral i{{background:var(--muted)}}
.fx-chip{{display:inline-flex;align-items:center;border-radius:999px;background:var(--brand-subtle);color:var(--brand-on);font-size:12px;font-weight:600;padding:3px 11px;margin:0 6px 6px 0}}
.fx-table{{width:100%;border-collapse:collapse}}.fx-table th{{text-align:left;color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;padding:8px 12px;border-bottom:1px solid var(--border)}}
.fx-table td{{padding:13px 12px;border-bottom:1px solid var(--border);font-size:14px;color:var(--text-2);vertical-align:middle}}.fx-table tr:last-child td{{border-bottom:0}}
.fx-table td.n .a{{font-weight:600;color:var(--text)}}.fx-table td.n .b{{font-size:12px;color:var(--muted);margin-top:2px}}
.fx-bar{{height:7px;border-radius:999px;background:var(--ring-track);overflow:hidden;width:110px;display:inline-block;vertical-align:middle;margin-right:10px}}.fx-bar>b{{display:block;height:100%;border-radius:999px}}
.fx-ring{{width:150px;height:150px;border-radius:50%;background:conic-gradient(var(--c,var(--ok)) calc(var(--p)*1%),var(--ring-track) 0);display:flex;align-items:center;justify-content:center;margin:0 auto}}
.fx-ring .in{{width:116px;height:116px;border-radius:50%;background:var(--surface-solid);display:flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:inset 0 2px 8px rgba(91,75,234,.08)}}
.fx-ring .in b{{font-size:26px;font-weight:800;color:var(--text);line-height:30px}}.fx-ring .in span{{font-size:11px;color:var(--muted);font-weight:600}}
.fx-check{{width:84px;height:84px;border-radius:50%;background:var(--ok-bg);display:flex;align-items:center;justify-content:center;margin:0 auto;box-shadow:0 0 0 10px rgba(34,197,94,.08)}}
.fx-check>div{{width:56px;height:56px;border-radius:50%;background:var(--ok);display:flex;align-items:center;justify-content:center;box-shadow:0 8px 18px rgba(34,197,94,.35)}}
.fx-steps{{display:flex;flex-wrap:wrap;align-items:center;gap:12px 24px;margin:14px 0 20px}}.fx-s{{display:flex;align-items:center;gap:9px;white-space:nowrap;font-size:13px;font-weight:600;color:var(--muted)}}
.fx-s .c{{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex:none;border:1.5px solid var(--border-strong);color:var(--muted);background:var(--surface-solid)}}
.fx-s.done{{color:var(--text-2)}}.fx-s.done .c{{background:var(--ok);border-color:var(--ok);color:#fff}}.fx-s.on{{color:var(--text)}}.fx-s.on .c{{background:var(--brand-grad);border-color:transparent;color:#fff;box-shadow:0 4px 10px rgba(91,75,234,.35)}}
.fx-list{{display:flex;flex-direction:column}}.fx-li{{display:flex;align-items:center;gap:14px;padding:13px 4px;border-bottom:1px solid var(--border);font-size:14px;color:var(--text)}}.fx-li:last-child{{border-bottom:0}}
.fx-li .no{{width:22px;color:var(--muted);font-size:13px}}.fx-li .tx{{flex:1}}
.fx-cfg{{display:flex;align-items:center;gap:12px;padding:12px 6px;border-bottom:1px solid var(--border)}}.fx-cfg:last-child{{border-bottom:0}}
.fx-cfg .ic{{width:34px;height:34px;border-radius:10px;background:var(--brand-subtle);color:var(--brand);display:flex;align-items:center;justify-content:center;flex:none}}
.fx-cfg .a{{font-size:12px;color:var(--muted)}}.fx-cfg .b{{font-size:13px;font-weight:600;color:var(--text)}}
.fx-metric{{display:flex;justify-content:space-between;font-size:13px;padding:5px 0;color:var(--text-2)}}.fx-metric b{{color:var(--text)}}
.fx-code{{font-family:"Roboto Mono",Consolas,monospace;font-size:12.5px;line-height:20px;color:var(--text-2);white-space:pre-wrap;background:var(--subtle);border-radius:12px;padding:16px}}
.fx-stage{{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:12px}}.fx-stage.run{{background:var(--brand-subtle)}}
.fx-stage .dot{{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex:none;color:#fff}}
.fx-stage .tx{{flex:1;min-width:0}}.fx-stage .a{{font-size:14px;font-weight:600;color:var(--text)}}.fx-stage .b{{font-size:12px;color:var(--muted)}}.fx-stage .d{{font-size:12px;font-weight:600;color:var(--muted)}}.fx-stage.pend .a{{color:var(--muted)}}
.fx-log{{background:#1B1A3F;border-radius:14px;padding:16px 20px;color:#CFCDF5;font-family:"Roboto Mono",Consolas,monospace;font-size:12px;line-height:18px;white-space:pre-wrap}}
.fx-alert{{display:flex;gap:12px;align-items:flex-start;padding:12px 16px;border-radius:12px;font-size:14px}}.fx-alert.warn{{background:var(--warn-bg);color:var(--warn-fg)}}.fx-alert.bad{{background:var(--bad-bg);color:var(--bad-fg)}}.fx-alert.info{{background:var(--info-bg);color:var(--info-fg)}}.fx-alert.ok{{background:var(--ok-bg);color:var(--ok-fg)}}
.fx-note{{display:flex;gap:10px;align-items:flex-start;font-size:12px;line-height:18px;color:var(--text-2);margin:10px 0}}
.fx-item{{background:var(--subtle);border-radius:12px;padding:12px 14px;margin-top:10px}}
.fx-logo{{display:flex;align-items:center;gap:10px;padding:2px 6px 22px;font-size:19px;font-weight:800;color:var(--text);letter-spacing:-.3px}}
.fx-logo .m{{width:34px;height:34px;border-radius:11px;background:var(--brand-grad);display:flex;align-items:center;justify-content:center;box-shadow:0 6px 14px rgba(91,75,234,.35)}}
.fx-model{{background:var(--surface-solid);border:1px solid var(--border);border-radius:14px;padding:12px 14px;font-size:12px;color:var(--muted);line-height:18px}}.fx-model b{{color:var(--text);font-weight:600}}
.fx-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}}
.fx-file{{display:flex;align-items:center;gap:12px;padding:10px 4px;border-bottom:1px solid var(--border)}}.fx-file .ic{{width:32px;height:32px;border-radius:10px;background:var(--brand-subtle);display:flex;align-items:center;justify-content:center;color:var(--brand);flex:none}}
.fx-file .nm{{flex:1;min-width:0}}.fx-file .a{{font-size:14px;font-weight:600;color:var(--text)}}.fx-file .b{{font-size:12px;color:var(--muted)}}
.fx-src{{border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin-top:8px;font-size:12px;color:var(--text-2);line-height:18px;background:var(--surface-solid)}}.fx-src b{{color:var(--text);font-weight:700}}
.fx-bub{{max-width:78%;padding:12px 16px;border-radius:16px;font-size:14px;line-height:21px;margin:8px 0;color:var(--text)}}.fx-bub.u{{background:var(--brand-subtle);margin-left:auto;border-bottom-right-radius:4px}}.fx-bub.a{{background:var(--surface-solid);border:1px solid var(--border);border-bottom-left-radius:4px}}
.fx-lang{{display:inline-flex;align-items:center;gap:10px;border:1px solid var(--border);border-radius:12px;padding:8px 12px;margin:0 10px 10px 0;font-size:14px;color:var(--text);background:var(--surface-solid)}}.fx-lang span{{font-size:12px;color:var(--muted)}}
.fx-iter{{display:flex;align-items:center;gap:14px;margin-top:8px}}.fx-iter .n{{min-width:92px;text-align:center;padding:10px 14px;border-radius:14px;border:1.5px dashed var(--border-strong);font-size:12px;color:var(--muted)}}.fx-iter .n b{{display:block;font-size:18px;color:var(--text)}}.fx-iter .n.on{{background:var(--brand-grad);border:0;color:#fff;box-shadow:0 8px 18px rgba(91,75,234,.30)}}.fx-iter .n.on b{{color:#fff}}.fx-iter .n.bad{{background:var(--bad-bg);border-color:var(--bad);color:var(--bad-fg)}}.fx-iter .n.bad b{{color:var(--bad-fg)}}
@media (max-width:1000px){{[data-testid="stHorizontalBlock"]{{flex-wrap:wrap !important}}[data-testid="stColumn"]{{min-width:min(100%,340px) !important}}}}
@media (max-width:1100px){{.fx-g4,.fx-g5,.fx-g3{{grid-template-columns:repeat(2,1fr)}}.fx-hero svg{{display:none}}}}


@property --p{{syntax:'<number>';inherits:false;initial-value:0}}
@keyframes fx-up{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
@keyframes fx-fade{{from{{opacity:0}}to{{opacity:1}}}}
@keyframes fx-pop{{0%{{transform:scale(.6);opacity:0}}70%{{transform:scale(1.08)}}100%{{transform:scale(1);opacity:1}}}}
@keyframes fx-ring{{from{{--p:0}}}}
@keyframes fx-grow{{from{{transform:scaleX(0)}}}}
@keyframes fx-draw{{to{{stroke-dashoffset:0}}}}
@keyframes fx-float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-9px)}}}}
@keyframes fx-spin{{to{{transform:rotate(360deg)}}}}
@keyframes fx-pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(59,130,246,.55)}}60%{{box-shadow:0 0 0 7px rgba(59,130,246,0)}}}}
@keyframes fx-shine{{from{{background-position:-200% 0}}to{{background-position:200% 0}}}}
.block-container [data-testid="stElementContainer"],.block-container [data-testid="stVerticalBlockBorderWrapper"]{{animation:fx-up .55s cubic-bezier(.2,.7,.2,1) both}}
.block-container [data-testid="stElementContainer"]:nth-child(2){{animation-delay:.05s}}.block-container [data-testid="stElementContainer"]:nth-child(3){{animation-delay:.1s}}
.block-container [data-testid="stElementContainer"]:nth-child(4){{animation-delay:.15s}}.block-container [data-testid="stElementContainer"]:nth-child(5){{animation-delay:.2s}}
.block-container [data-testid="stElementContainer"]:nth-child(n+6){{animation-delay:.25s}}
.fx-card{{transition:transform .25s ease,box-shadow .25s ease}}
.fx-tile:hover,.fx-kpi:hover{{transform:translateY(-4px);box-shadow:0 18px 38px rgba(91,75,234,.18)}}
.fx-item,.fx-src,.fx-lang,.fx-file{{transition:transform .2s ease,background .2s ease}}.fx-item:hover,.fx-lang:hover{{transform:translateY(-2px)}}.fx-file:hover{{background:var(--subtle);border-radius:10px}}
.fx-table tbody tr{{animation:fx-fade .5s ease both;transition:background .2s}}.fx-table tbody tr:hover{{background:var(--subtle)}}
.fx-table tbody tr:nth-child(2){{animation-delay:.06s}}.fx-table tbody tr:nth-child(3){{animation-delay:.12s}}.fx-table tbody tr:nth-child(4){{animation-delay:.18s}}.fx-table tbody tr:nth-child(5){{animation-delay:.24s}}.fx-table tbody tr:nth-child(n+6){{animation-delay:.3s}}
.fx-ring{{animation:fx-ring 1.2s cubic-bezier(.2,.7,.2,1) both}}
.fx-bar>b{{transform-origin:left;animation:fx-grow .9s cubic-bezier(.2,.7,.2,1) both}}
svg polyline{{stroke-dasharray:2400;stroke-dashoffset:2400;animation:fx-draw 1.5s ease .2s forwards}}
.fx-float{{animation:fx-float 5s ease-in-out infinite}}
.fx-check{{animation:fx-pop .65s cubic-bezier(.2,.9,.3,1.2) both}}
.fx-stage.run .dot svg{{animation:fx-spin 1.3s linear infinite}}.fx-stage.run{{background:linear-gradient(100deg,var(--brand-subtle) 30%,rgba(255,255,255,.7) 50%,var(--brand-subtle) 70%);background-size:200% 100%;animation:fx-shine 2.2s linear infinite}}
.fx-info i{{animation:fx-pulse 1.7s ease-out infinite}}
.fx-bub{{animation:fx-up .35s ease both}}.fx-alert,.fx-note{{animation:fx-fade .5s ease both}}
.fx-iter .n{{animation:fx-pop .5s ease both}}.fx-iter .n:nth-child(3){{animation-delay:.15s}}.fx-iter .n:nth-child(5){{animation-delay:.3s}}.fx-iter .n:nth-child(7){{animation-delay:.45s}}
[data-baseweb="tab-panel"]{{animation:fx-fade .4s ease both}}
[data-baseweb="tab"]{{transition:background .2s,border-color .2s,transform .15s}}[data-baseweb="tab"]:hover{{transform:translateY(-1px)}}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-secondary"]{{transition:transform .15s ease,box-shadow .2s ease,filter .2s ease,background .2s}}
[data-testid="stBaseButton-primary"]:hover{{transform:translateY(-2px)}}[data-testid="stBaseButton-primary"]:active,[data-testid="stBaseButton-secondary"]:active{{transform:scale(.97)}}
[data-testid="stSidebar"] .stButton button{{transition:background .2s ease,transform .2s ease}}[data-testid="stSidebar"] .stButton button:hover{{transform:translateX(4px)}}
.fx-top .ic,.fx-top .av{{transition:transform .2s}}.fx-top .ic:hover,.fx-top .av:hover{{transform:scale(1.1)}}
@media (prefers-reduced-motion:reduce){{*,*::before,*::after{{animation:none !important;transition:none !important}}svg polyline{{stroke-dashoffset:0}}}}
</style>"""


# ------------------------------------------------------------------------------------------------ components
STATUS = {"succeeded": ("ok", "Completed"), "succeeded_with_warnings": ("warn", "With warnings"), "blocked": ("bad", "Blocked by guard"), "failed": ("bad", "Failed"),
          "running": ("info", "Running"), "queued": ("info", "Queued"), "interrupted": ("warn", "Interrupted"), "finished": ("neutral", "Finished"), "created": ("neutral", "Draft")}


def badge(kind: str, label: str) -> str:
    return f'<span class="fx-badge fx-{kind}"><i></i>{esc(str(label))}</span>'


def status_badge(status: str) -> str:
    kind, label = STATUS.get(status, ("neutral", status.replace("_", " ").title()))
    return badge(kind, label)


def chip(text: str) -> str:
    return f'<span class="fx-chip">{esc(text)}</span>'


def topbar(initial: str = "U") -> str:
    return f'<div class="fx-top"><div class="ic">{icon("bell", 18)}</div><div class="av">{esc(initial)}</div></div>'


def header(title: str, subtitle: str = "") -> str:
    return f'<h1 class="fx-h1">{esc(title)}</h1>' + (f'<p class="fx-sub">{esc(subtitle)}</p>' if subtitle else "")


def hero_art() -> str:
    """Layered translucent plates (the 'agent factory' stack) as inline SVG."""
    return ('<svg class="fx-float" width="300" height="190" viewBox="0 0 300 190" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs>'
            '<linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#C7BFFF"/><stop offset="1" stop-color="#7C6CF7"/></linearGradient>'
            '<linearGradient id="g2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#E3DEFF"/><stop offset="1" stop-color="#9F94FA"/></linearGradient>'
            '<linearGradient id="g3" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#BDE0FF"/><stop offset="1" stop-color="#6C8CF7"/></linearGradient>'
            '<filter id="b"><feGaussianBlur stdDeviation="12"/></filter></defs>'
            '<ellipse cx="150" cy="168" rx="96" ry="12" fill="#7C6CF7" opacity=".25" filter="url(#b)"/>'
            '<path d="M150 118 252 142 150 170 48 142z" fill="url(#g3)" opacity=".85"/>'
            '<path d="M150 88 252 112 150 140 48 112z" fill="url(#g2)" opacity=".92"/>'
            '<path d="M150 56 252 80 150 108 48 80z" fill="url(#g1)"/>'
            '<path d="M150 20 190 56 150 70 110 56z" fill="#fff" opacity=".55"/><circle cx="150" cy="46" r="9" fill="#fff" opacity=".9"/>'
            '<path d="m146 46 3 3 5-6" stroke="#6C5CE7" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def hero(title: str, tagline: str, body: str) -> str:
    return (f'<div class="fx-card fx-hero"><div class="t"><div class="w">{esc(title)}</div><div class="g">{esc(tagline)}</div>'
            f'<div class="fx-sub" style="max-width:440px">{esc(body)}</div></div>{hero_art()}</div>')


def tile(icon_name: str, value: str, label: str, kind: str = "brand") -> str:
    bg, fg = {"brand": ("var(--brand-subtle)", "var(--brand)"), "ok": ("var(--ok-bg)", "var(--ok)"), "info": ("var(--info-bg)", "var(--info)"), "warn": ("var(--warn-bg)", "var(--warn)")}[kind]
    return f'<div class="fx-card fx-tile"><div class="ic" style="background:{bg};color:{fg}">{icon(icon_name, 22)}</div><div><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div></div>'


def tiles(items: list[tuple[str, str, str, str]], cols: int = 4) -> str:
    return f'<div class="fx-grid fx-g{cols}">' + "".join(tile(*i) for i in items) + "</div>"


def kpi_cards(items: list[tuple[str, str, str]], cols: int = 4) -> str:
    cells = "".join(f'<div class="fx-card fx-kpi"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div><div class="h">{esc(h)}</div></div>' for l, v, h in items)
    return f'<div class="fx-grid fx-g{cols}">{cells}</div>'


def ring(percent: float | None, label: str = "Overall Score") -> str:
    p = 0 if percent is None else max(0, min(100, percent))
    col = "var(--ok)" if p >= 90 else "var(--warn)" if p >= 70 else "var(--bad)"
    shown = "—" if percent is None else f"{percent:g}%"
    return f'<div class="fx-ring" style="--p:{p};--c:{col}"><div class="in"><b>{esc(shown)}</b><span>{esc(label)}</span></div></div>'


def big_check(ok: bool = True) -> str:
    col, bg = ("var(--ok)", "var(--ok-bg)") if ok else ("var(--warn)", "var(--warn-bg)")
    return f'<div class="fx-check" style="background:{bg}"><div style="background:{col}">{icon("check" if ok else "alert", 30, "#fff", 2.8)}</div></div>'


def wizard_steps(steps: list[str], states: list[str]) -> str:
    """states: 'done' | 'on' | 'todo' per step (the live state of the run, not decoration)."""
    out = []
    for i, (name, st) in enumerate(zip(steps, states)):
        inner = icon("check", 14, "#fff", 3) if st == "done" else str(i + 1)
        out.append(f'<div class="fx-s {st}"><div class="c">{inner}</div>{esc(name)}</div>')
    return f'<div class="fx-steps">{"".join(out)}</div>'


def pct_bar(value: int | float | None, width: int = 110) -> str:
    if value is None:
        return '<span class="fx-muted">—</span>'
    col = "var(--ok)" if value >= 90 else "var(--warn)" if value >= 70 else "var(--bad)"
    return f'<span class="fx-bar" style="width:{width}px"><b style="width:{value}%;background:{col}"></b></span><b style="color:var(--text);font-weight:600">{value:g}%</b>'


def bar_chart(rows: list[tuple[str, int | None, int | None]]) -> str:
    groups = ""
    for name, dev, hold in rows:
        b = "".join(f'<b style="display:block;width:18px;border-radius:5px 5px 0 0;height:{max(2, int(v * 1.5))}px;background:var(--{c})" title="{esc(label)} {v}%"></b>' for v, c, label in ((dev or 0, "brand", "dev"), (hold or 0, "ok", "hold-out")) if v is not None)
        groups += f'<div style="display:flex;flex-direction:column;align-items:center;gap:8px;flex:1"><div style="display:flex;align-items:flex-end;gap:4px;height:170px">{b}</div><div class="fx-muted" style="text-align:center">{esc(name)}</div></div>'
    return f'<div style="display:flex;align-items:flex-end;justify-content:space-between;height:200px;gap:10px;margin-top:14px">{groups}</div>'


def legend() -> str:
    return '<div style="display:flex;gap:16px;font-size:12px;color:var(--text-2)"><span><i style="display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;background:var(--brand)"></i>Dev</span><span><i style="display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;background:var(--ok)"></i>Hold-out</span></div>'


def trend_chart(points: list[tuple[str, float | None]], width: int = 460, height: int = 190) -> str:
    """Line chart of scores (0-100) by label, drawn as SVG with grid lines and value dots."""
    pts = [(l, v) for l, v in points if v is not None]
    if not pts:
        return '<div class="fx-muted">No scores recorded.</div>'
    lo, hi = min(v for _, v in pts), max(v for _, v in pts)
    lo, hi = max(0, min(lo, hi) - 8), min(100, max(hi, lo + 5) + 4)
    px, py = 46, 14
    W, H = width - px - 12, height - py - 34
    xs = [px + (W * i / max(1, len(pts) - 1)) for i in range(len(pts))]
    ys = [py + H * (1 - (v - lo) / (hi - lo or 1)) for _, v in pts]
    grid = "".join(f'<line x1="{px}" x2="{width - 8}" y1="{py + H * k / 4:.1f}" y2="{py + H * k / 4:.1f}" stroke="var(--border)"/><text x="{px - 8}" y="{py + H * k / 4 + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{round(hi - (hi - lo) * k / 4)}%</text>' for k in range(5))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--surface-solid)" stroke="var(--brand)" stroke-width="2.5"/><text x="{x:.1f}" y="{height - 12}" text-anchor="middle" font-size="11" fill="var(--muted)">{esc(l)}</text>' for (l, _), x, y in zip(pts, xs, ys))
    return f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">{grid}<polyline points="{line}" fill="none" stroke="var(--brand)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>{dots}</svg>'


def table(headers: list[str], rows: list[list[str]], widths: list[str] | None = None) -> str:
    th = "".join(f'<th style="{"width:" + widths[i] if widths and widths[i] else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{' class=n' if i == 0 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table class="fx-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def name_cell(a: str, b: str = "") -> str:
    return f'<div class="a">{esc(a)}</div>' + (f'<div class="b">{esc(b)}</div>' if b else "")


def check_list(items: list[str], ok: list[bool] | None = None) -> str:
    rows = ""
    for i, t in enumerate(items):
        good = True if ok is None else ok[i]
        mark = icon("check", 18, "var(--ok)", 2.6) if good else icon("alert", 18, "var(--warn)", 2.2)
        rows += f'<div class="fx-li"><div class="no">{i + 1}</div><div class="tx">{esc(t)}</div>{mark}</div>'
    return f'<div class="fx-list">{rows}</div>'


def cfg_row(icon_name: str, label: str, value: str) -> str:
    return f'<div class="fx-cfg"><div class="ic">{icon(icon_name, 18)}</div><div><div class="a">{esc(label)}</div><div class="b">{esc(value)}</div></div></div>'


def metric(label: str, value: str) -> str:
    return f'<div class="fx-metric"><span>{esc(label)}</span><b>{esc(value)}</b></div>'


def code_block(text: str) -> str:
    return f'<div class="fx-code">{esc(text)}</div>'


def iter_chain(items: list[tuple[str, str, str]]) -> str:
    """items: (label, score text, state) with state in on | bad | ''."""
    out = []
    for i, (l, v, st) in enumerate(items):
        out.append(f'<div class="n {st}"><b>{esc(v)}</b>{esc(l)}</div>')
        if i < len(items) - 1:
            out.append(f'<div style="color:var(--brand)">{icon("arrow", 20)}</div>')
    return f'<div class="fx-iter">{"".join(out)}</div>'


def bubble(role: str, text: str, caption: str = "") -> str:
    cap = f'<div class="fx-muted" style="margin-top:6px">{esc(caption)}</div>' if caption else ""
    return f'<div class="fx-bub {"u" if role == "user" else "a"}">{esc(text)}{cap}</div>'


def stage(state: str, title: str, meta: str, dur: str) -> str:
    cfg = {"done": ("var(--ok)", icon("check", 14, "#fff", 2.5), ""), "run": ("var(--brand)", icon("optimize", 14, "#fff", 2), "run"), "pend": ("var(--subtle)", "", "pend"),
           "fail": ("var(--bad)", icon("x", 14, "#fff", 2.5), "")}[state]
    dot = f'<div class="dot" style="background:{cfg[0]};{"border:1.5px solid var(--border-strong)" if state == "pend" else ""}">{cfg[1]}</div>'
    return f'<div class="fx-stage {cfg[2]}">{dot}<div class="tx"><div class="a">{esc(title)}</div><div class="b">{esc(meta)}</div></div><div class="d">{esc(dur)}</div></div>'


def alert(kind: str, text: str, icon_name: str = "alert") -> str:
    col = {"warn": "var(--warn-fg)", "bad": "var(--bad-fg)", "info": "var(--info-fg)", "ok": "var(--ok-fg)"}[kind]
    return f'<div class="fx-alert {kind}">{icon(icon_name, 20, col)}<div>{esc(text)}</div></div>'


def note(ok: bool, text: str) -> str:
    return f'<div class="fx-note">{icon("ok" if ok else "info", 18, "var(--ok)" if ok else "var(--warn)")}<div>{esc(text)}</div></div>'


def log_block(lines: list[str]) -> str:
    return f'<div class="fx-log"><div class="fx-caption" style="color:#A9A7D9;margin-bottom:8px">Live log</div>{esc(chr(10).join(lines[-8:]) or "waiting for output…")}</div>'


def sidebar_logo() -> str:
    return f'<div class="fx-logo"><div class="m">{icon("bolt", 19, "#fff", 2)}</div>ForgeAI</div>'


def model_card(ok: bool, model: str, embed: str | None) -> str:
    col = "#22C55E" if ok else "#EF4444"
    return (f'<div class="fx-model"><div><span class="fx-dot" style="background:{col}"></span><b>{"Ollama connected" if ok else "Ollama not reachable"}</b></div>'
            f'<div style="margin-top:6px">{esc(model)}{" · " + esc(embed) if embed else ""}</div></div>')


def file_row(name: str, meta: str, lang_html: str, status_html: str) -> str:
    return (f'<div class="fx-file"><div class="ic">{icon("file", 16)}</div><div class="nm"><div class="a">{esc(name)}</div><div class="b">{esc(meta)}</div></div>'
            f'<div style="width:100px">{lang_html}</div><div style="width:170px">{status_html}</div></div>')


def demo_card() -> str:
    return ('<div class="fx-model"><div><span class="fx-dot" style="background:#6C5CE7"></span><b>Demo mode</b></div>'
            '<div style="margin-top:6px">Read-only: real builds and results, no live model.</div></div>')
