import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import io, os, base64, time
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

st.set_page_config(page_title="GraphAISMNet", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Read page from URL query params ──────────────────────────────────────────
params   = st.query_params
PAGE     = params.get("page", "home")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="block-container"]{
    background:#F0F4FA!important;color:#1A2744!important;font-family:'Inter',sans-serif!important;}
[data-testid="block-container"]{padding:0!important;max-width:100%!important;}
#MainMenu,header,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}

/* ── NAVBAR ─────────────────────────────────────────────────────────────── */
.navbar{
    background:linear-gradient(135deg,#0D2E6E 0%,#1B4FA8 55%,#0D6B4F 100%);
    padding:0 2.5rem;display:flex;align-items:center;justify-content:space-between;
    height:68px;box-shadow:0 3px 20px rgba(13,46,110,.35);
    position:sticky;top:0;z-index:1000;}
.navbar-brand{display:flex;align-items:center;gap:.8rem;}
.navbar-logo{width:40px;height:40px;background:rgba(255,255,255,.15);border-radius:10px;
    display:flex;align-items:center;justify-content:center;font-size:1.3rem;}
.navbar-title{font-family:'Poppins',sans-serif;font-weight:700;font-size:1.3rem;color:white;letter-spacing:-.01em;}
.navbar-title span{color:#5DDEAE;}
.navbar-subtitle{font-size:.7rem;color:rgba(255,255,255,.6);font-weight:400;display:block;margin-top:-2px;}
.navbar-links{display:flex;gap:.2rem;}
.nav-btn{
    color:rgba(255,255,255,.8);background:transparent;border:none;
    padding:.42rem .95rem;border-radius:6px;font-size:.86rem;font-weight:500;
    text-decoration:none;display:inline-block;transition:all .2s;font-family:'Inter';}
.nav-btn:hover{background:rgba(255,255,255,.15);color:white;}
.nav-btn.active{background:rgba(255,255,255,.18);color:white;
    border-bottom:2px solid #5DDEAE;border-radius:6px 6px 0 0;}

/* ── HERO ────────────────────────────────────────────────────────────────── */
.hero-banner{
    background:linear-gradient(135deg,#0D2E6E 0%,#1B4FA8 60%,#0D6B4F 100%);
    padding:2.5rem 3rem 2rem;color:white;text-align:center;position:relative;overflow:hidden;}
.hero-banner::before{content:'';position:absolute;inset:0;
    background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");}
.hero-title{font-family:'Poppins',sans-serif;font-weight:800;font-size:2.4rem;
    letter-spacing:-.02em;margin-bottom:.4rem;}
.hero-title span{color:#5DDEAE;}
.hero-sub{font-size:1.05rem;color:rgba(255,255,255,.75);margin-bottom:1.2rem;font-weight:300;}
.hero-badges{display:flex;justify-content:center;gap:.8rem;flex-wrap:wrap;}
.hero-badge{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
    border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:rgba(255,255,255,.9);font-weight:500;}
.hero-badge.green{background:rgba(93,222,174,.2);border-color:rgba(93,222,174,.4);color:#5DDEAE;}

/* ── PAGE SECTION ────────────────────────────────────────────────────────── */
.page-section{padding:2rem 2.5rem;}
.section-heading{font-family:'Poppins',sans-serif;font-weight:700;font-size:1.5rem;
    color:#0D2E6E;margin-bottom:.3rem;}
.section-sub{font-size:.9rem;color:#6B7A99;margin-bottom:1.5rem;}

/* ── CARDS ───────────────────────────────────────────────────────────────── */
.card{background:white;border-radius:14px;padding:1.5rem;
    box-shadow:0 2px 16px rgba(13,46,110,.08);border:1px solid rgba(13,46,110,.07);}
.card-blue{border-top:4px solid #1B4FA8;}
.card-green{border-top:4px solid #0D7A5F;}

/* ── STAT PILLS ──────────────────────────────────────────────────────────── */
.stat-grid{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0;}
.stat-pill{background:#F0F4FA;border:1px solid #D5E0F5;border-radius:10px;
    padding:.8rem 1.2rem;text-align:center;flex:1;min-width:90px;}
.stat-val{font-family:'Poppins';font-weight:700;font-size:1.5rem;color:#1B4FA8;}
.stat-val.green{color:#0D7A5F;}
.stat-lbl{font-size:.72rem;color:#6B7A99;text-transform:uppercase;letter-spacing:.06em;}

/* ── BUTTONS ─────────────────────────────────────────────────────────────── */
.stButton>button{
    background:linear-gradient(135deg,#1B4FA8 0%,#0D7A5F 100%)!important;
    color:white!important;border:none!important;border-radius:8px!important;
    padding:.65rem 2rem!important;font-family:'Poppins'!important;font-weight:600!important;
    font-size:.92rem!important;width:100%!important;}
.stButton>button:hover{opacity:.88!important;}

/* ── INPUTS ──────────────────────────────────────────────────────────────── */
.stTextArea textarea,.stTextInput input{
    background:white!important;border:1.5px solid #D5E0F5!important;
    border-radius:8px!important;color:#1A2744!important;
    font-family:'JetBrains Mono',monospace!important;font-size:.88rem!important;}
.stTextArea textarea:focus,.stTextInput input:focus{
    border-color:#1B4FA8!important;box-shadow:0 0 0 3px rgba(27,79,168,.12)!important;}
label{color:#3A4A6B!important;font-weight:500!important;font-size:.88rem!important;}

.stSelectbox>div>div{background:white!important;border:1.5px solid #D5E0F5!important;
    border-radius:8px!important;color:#1A2744!important;}

/* ── RESULT ──────────────────────────────────────────────────────────────── */
.result-active{background:linear-gradient(135deg,#E8F8F2,#D0F0E4);border:2px solid #0D7A5F;
    border-radius:12px;padding:1.4rem 1.8rem;text-align:center;margin:1rem 0;}
.result-inactive{background:linear-gradient(135deg,#FEF0F0,#FDD8D8);border:2px solid #C0392B;
    border-radius:12px;padding:1.4rem 1.8rem;text-align:center;margin:1rem 0;}
.result-label{font-family:'Poppins';font-weight:700;font-size:1.3rem;margin-bottom:.3rem;}
.result-sub{font-size:.88rem;color:#4A5568;}

.conf-wrap{margin:.8rem 0;}
.conf-row{display:flex;justify-content:space-between;font-size:.8rem;color:#6B7A99;margin-bottom:.3rem;}
.conf-track{height:8px;background:#E8EEF8;border-radius:4px;overflow:hidden;}
.conf-fill{height:100%;border-radius:4px;}

.desc-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-top:1rem;}
.desc-item{background:#F7FAFF;border:1px solid #E0E8F5;border-radius:8px;padding:.7rem .9rem;}
.desc-key{font-size:.68rem;color:#8A9ABB;text-transform:uppercase;letter-spacing:.06em;}
.desc-val{font-family:'Poppins';font-weight:600;font-size:1rem;color:#1B4FA8;margin-top:.1rem;}

/* ── TABLE ───────────────────────────────────────────────────────────────── */
.perf-table{width:100%;border-collapse:collapse;font-size:.88rem;}
.perf-table th{background:#0D2E6E;color:white;padding:.7rem 1rem;text-align:left;
    font-family:'Poppins';font-weight:600;font-size:.8rem;letter-spacing:.04em;}
.perf-table td{padding:.7rem 1rem;border-bottom:1px solid #E8EEF8;color:#2D3748;}
.perf-table tr:hover td{background:#F0F4FA;}
.perf-table .best{color:#0D7A5F;font-weight:700;}
.perf-table .model-col{font-family:'Poppins';font-weight:600;color:#0D2E6E;}

/* ── TABS ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:white!important;border-radius:10px 10px 0 0!important;
    border-bottom:2px solid #E0E8F5!important;gap:0!important;padding:0 1rem!important;}
.stTabs [data-baseweb="tab"]{font-family:'Poppins'!important;font-weight:500!important;
    color:#6B7A99!important;font-size:.9rem!important;padding:.8rem 1.5rem!important;
    border-radius:0!important;background:transparent!important;}
.stTabs [aria-selected="true"]{color:#1B4FA8!important;border-bottom:3px solid #1B4FA8!important;font-weight:600!important;}
.stTabs [data-baseweb="tab-panel"]{background:white!important;border:1px solid #E0E8F5!important;
    border-top:none!important;border-radius:0 0 10px 10px!important;padding:1.5rem!important;}

[data-testid="stFileUploader"]{background:#F7FAFF!important;border:2px dashed #B8CCF0!important;border-radius:10px!important;}
[data-testid="stDataFrame"]{border:1px solid #E0E8F5!important;border-radius:8px!important;}
[data-testid="stSidebar"]{background:white!important;border-right:1px solid #E0E8F5!important;}
[data-testid="stDownloadButton"] button{background:#F0F4FA!important;border:1.5px solid #1B4FA8!important;color:#1B4FA8!important;border-radius:8px!important;}

.info-box{background:#EEF4FF;border-left:4px solid #1B4FA8;border-radius:0 8px 8px 0;
    padding:.8rem 1rem;font-size:.87rem;color:#2D3748;margin:.5rem 0;}
.warn-box{background:#FFF8E6;border-left:4px solid #F5A623;border-radius:0 8px 8px 0;
    padding:.8rem 1rem;font-size:.87rem;color:#5A3E00;margin:.5rem 0;}

.site-footer{background:#0D2E6E;color:rgba(255,255,255,.7);text-align:center;
    padding:1.5rem;font-size:.82rem;margin-top:3rem;}
.site-footer a{color:#5DDEAE;text-decoration:none;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-thumb{background:#B8CCF0;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NODE_DIM=21; DESC_DIM=13; THRESHOLD=6.0
Y_REG_MEAN=6.408  # UPDATE with Cell 4 value
Y_REG_STD =1.208   # UPDATE with Cell 4 value

# ── Model classes ─────────────────────────────────────────────────────────────
class SimpleGAT(nn.Module):
    def __init__(self,node_dim=21,desc_dim=13,hidden=128,heads=4,dropout=.2,num_layers=4,**kw):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(node_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout))
        self.eproj=nn.Linear(7,32)
        self.convs=nn.ModuleList([GATv2Conv(hidden,hidden,heads=heads,concat=False,edge_dim=32) for _ in range(num_layers)])
        self.norms=nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.desc_bn=nn.BatchNorm1d(desc_dim); p=hidden*2+desc_dim
        self.cls_head=nn.Sequential(nn.Linear(p,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(64,1))
        self.reg_head=nn.Sequential(nn.Linear(p,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(64,1))
    def forward(self,data):
        x=self.embed(data.x); ea=self.eproj(data.edge_attr)
        for conv,norm in zip(self.convs,self.norms): x=x+norm(F.gelu(conv(x,data.edge_index,ea)))
        desc=data.desc
        if desc.size(0)>1: desc=self.desc_bn(desc)
        g=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch),desc],-1)
        return self.cls_head(g).squeeze(-1),self.reg_head(g).squeeze(-1)

class SimpleGT(nn.Module):
    def __init__(self,node_dim=21,desc_dim=13,hidden=128,heads=4,dropout=.2,num_layers=4,**kw):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(node_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout))
        self.eproj=nn.Linear(7,32)
        self.convs=nn.ModuleList([TransformerConv(hidden,hidden,heads=heads,concat=False,edge_dim=32) for _ in range(num_layers)])
        self.norms=nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.desc_bn=nn.BatchNorm1d(desc_dim); p=hidden*2+desc_dim
        self.cls_head=nn.Sequential(nn.Linear(p,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(64,1))
        self.reg_head=nn.Sequential(nn.Linear(p,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(64,1))
    def forward(self,data):
        x=self.embed(data.x); ea=self.eproj(data.edge_attr)
        for conv,norm in zip(self.convs,self.norms): x=x+norm(F.gelu(conv(x,data.edge_index,ea)))
        desc=data.desc
        if desc.size(0)>1: desc=self.desc_bn(desc)
        g=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch),desc],-1)
        return self.cls_head(g).squeeze(-1),self.reg_head(g).squeeze(-1)

# ── Feature functions ─────────────────────────────────────────────────────────
def atom_features(atom):
    atom_types=['C','N','O','S','F','Cl','Br','I','P','B']
    f=[1. if atom.GetSymbol()==t else 0. for t in atom_types]
    f+=[atom.GetAtomicNum()/100.,atom.GetDegree()/8.,atom.GetFormalCharge()/5.,
        atom.GetTotalNumHs()/8.,atom.GetTotalValence()/8.,float(atom.GetIsAromatic()),
        float(atom.IsInRing()),float(atom.GetChiralTag()!=Chem.rdchem.ChiralType.CHI_UNSPECIFIED)]
    hyb=atom.GetHybridization()
    for ht in [Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3]:
        f.append(1. if hyb==ht else 0.)
    return f

def bond_features(bond):
    bt=bond.GetBondType()
    return [float(bt==Chem.rdchem.BondType.SINGLE),float(bt==Chem.rdchem.BondType.DOUBLE),
            float(bt==Chem.rdchem.BondType.TRIPLE),float(bt==Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),float(bond.IsInRing()),
            float(bond.GetStereo()!=Chem.rdchem.BondStereo.STEREONONE)]

def smiles_to_graph(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x=[atom_features(a) for a in mol.GetAtoms()]
    ei,ea=[],[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        bf=bond_features(bond); ei+=[[i,j],[j,i]]; ea+=[bf,bf]
    if not ei: ei=[[0,0],[0,0]]; ea=[[0.]*7,[0.]*7]
    try:
        desc=[Descriptors.MolWt(mol)/1000.,Descriptors.MolLogP(mol)/10.,
              Descriptors.TPSA(mol)/200.,Descriptors.NumRotatableBonds(mol)/20.,
              QED.qed(mol),Descriptors.NumHDonors(mol)/10.,Descriptors.NumHAcceptors(mol)/10.,
              float(rdMolDescriptors.CalcNumAromaticRings(mol))/5.,Descriptors.FractionCSP3(mol),
              float(mol.GetNumHeavyAtoms())/50.,float(rdMolDescriptors.CalcNumRings(mol))/10.,
              min(Descriptors.BertzCT(mol)/1000.,3.),float(rdMolDescriptors.CalcNumHeteroatoms(mol))/20.]
    except: desc=[0.]*13
    g=Data(x=torch.tensor(x,dtype=torch.float),
           edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
           edge_attr=torch.tensor(ea,dtype=torch.float),
           desc=torch.tensor(desc,dtype=torch.float).view(1,-1),
           y_cls=torch.zeros(1),y_reg=torch.zeros(1))
    g.batch=torch.zeros(g.x.size(0),dtype=torch.long)
    return g

def mol_to_svg(smiles,size=(420,280)):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    AllChem.Compute2DCoords(mol)
    drawer=rdMolDraw2D.MolDraw2DSVG(size[0],size[1])
    drawer.drawOptions().backgroundColour=(0.95,0.97,1.0,1.0)
    drawer.DrawMolecule(mol); drawer.FinishDrawing()
    return drawer.GetDrawingText()

def compute_props(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return {}
    try:
        return {"Mol. Weight":f"{Descriptors.MolWt(mol):.2f} Da",
                "LogP":f"{Descriptors.MolLogP(mol):.3f}",
                "TPSA":f"{Descriptors.TPSA(mol):.2f} Å²",
                "HB Donors":int(Descriptors.NumHDonors(mol)),
                "HB Acceptors":int(Descriptors.NumHAcceptors(mol)),
                "Rotatable Bonds":int(Descriptors.NumRotatableBonds(mol)),
                "QED":f"{QED.qed(mol):.4f}",
                "Aromatic Rings":int(rdMolDescriptors.CalcNumAromaticRings(mol)),
                "Heavy Atoms":int(mol.GetNumHeavyAtoms())}
    except: return {}

@st.cache_resource
def load_models():
    gt=SimpleGT(); gt_ok=False
    gat=SimpleGAT(); gat_ok=False
    if os.path.exists("model_random_gt.pt"):
        gt.load_state_dict(torch.load("model_random_gt.pt",map_location="cpu")); gt_ok=True
    if os.path.exists("model_scaffold_gat.pt"):
        gat.load_state_dict(torch.load("model_scaffold_gat.pt",map_location="cpu")); gat_ok=True
    gt.eval(); gat.eval()
    return gt,gt_ok,gat,gat_ok

gt_model,gt_loaded,gat_model,gat_loaded=load_models()

def run_predict(model,smiles):
    g=smiles_to_graph(smiles.strip())
    if g is None: return None
    model.eval()
    with torch.no_grad():
        co,ro=model(g)
        prob=float(torch.sigmoid(co).item())
        pic50=float(ro.item())*Y_REG_STD+Y_REG_MEAN
    return {"prob":prob,"pic50":round(pic50,4),"active":prob>=0.5,
            "ic50":round(10**(6-pic50),4) if pic50<12 else ">1M"}

EXAMPLES={
    "Ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Celecoxib":   "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(N)cc2)cc1S(N)(=O)=O",
    "Naproxen":    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
    "Diclofenac":  "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Indomethacin":"COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
}

# ── Helper: result block ───────────────────────────────────────────────────────
def show_result(r):
    css="result-active" if r["active"] else "result-inactive"
    color="#0D7A5F" if r["active"] else "#C0392B"
    icon="✅" if r["active"] else "❌"
    bar_bg="linear-gradient(90deg,#0D7A5F,#5DDEAE)" if r["active"] else "linear-gradient(90deg,#C0392B,#E74C3C)"
    st.markdown(f"""
    <div class="{css}">
        <div class="result-label" style="color:{color};">{icon} {"ACTIVE" if r["active"] else "INACTIVE"}</div>
        <div class="result-sub">Confidence: {r["prob"]*100:.1f}% &nbsp;·&nbsp; Threshold: pIC₅₀ ≥ 6.0</div>
    </div>
    <div class="conf-wrap">
        <div class="conf-row"><span>Activity Probability</span><span>{r["prob"]:.4f}</span></div>
        <div class="conf-track"><div class="conf-fill" style="width:{r["prob"]*100:.1f}%;background:{bar_bg};"></div></div>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(f'<div class="stat-pill"><div class="stat-val">{r["pic50"]}</div><div class="stat-lbl">pIC₅₀</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-pill"><div class="stat-val" style="font-size:1rem;">{r["ic50"]}</div><div class="stat-lbl">IC₅₀ (µM)</div></div>',unsafe_allow_html=True)
    with c3:
        tm="✓ Met" if r["pic50"]>=THRESHOLD else "✗ Not Met"
        tc="#0D7A5F" if r["pic50"]>=THRESHOLD else "#C0392B"
        st.markdown(f'<div class="stat-pill"><div class="stat-val" style="font-size:.95rem;color:{tc};">{tm}</div><div class="stat-lbl">Threshold ≥6.0</div></div>',unsafe_allow_html=True)
    props=compute_props(st.session_state.get("last_smi",""))
    if props:
        items="".join([f'<div class="desc-item"><div class="desc-key">{k}</div><div class="desc-val">{v}</div></div>' for k,v in props.items()])
        st.markdown(f'<div style="margin-top:1rem;"><div style="font-family:Poppins;font-weight:600;font-size:.85rem;color:#0D2E6E;margin-bottom:.5rem;">Physicochemical Properties</div><div class="desc-grid">{items}</div></div>',unsafe_allow_html=True)

# ── Prediction panel (shared) ──────────────────────────────────────────────────
def predict_panel(model, model_name, btn_label, smi_key, btn_key):
    left,right=st.columns([1,1.2],gap="large")
    with left:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:Poppins;font-weight:600;font-size:1rem;color:#0D2E6E;margin-bottom:.8rem;">🔬 Single Molecule Input</div>',unsafe_allow_html=True)
        ex=st.selectbox("Load example",["— Custom input —"]+list(EXAMPLES.keys()),key=smi_key+"_ex")
        default=EXAMPLES[ex] if ex!="— Custom input —" else ""
        smi=st.text_area("SMILES string",value=default,height=90,placeholder="Paste SMILES here…",key=smi_key)
        st.session_state["last_smi"]=smi
        run=st.button(btn_label,key=btn_key)
        if smi.strip():
            mol=Chem.MolFromSmiles(smi.strip())
            if mol:
                svg=mol_to_svg(smi.strip(),(400,250))
                if svg:
                    b64=base64.b64encode(svg.encode()).decode()
                    st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;border-radius:8px;margin-top:.8rem;"/>',unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">⚠️ Invalid SMILES</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div style="font-family:Poppins;font-weight:600;font-size:1rem;color:#0D2E6E;margin-bottom:.8rem;">📊 Prediction Results</div>',unsafe_allow_html=True)
        if run and smi.strip():
            mol=Chem.MolFromSmiles(smi.strip())
            if not mol: st.error("Invalid SMILES.")
            else:
                with st.spinner(f"Running {model_name} inference…"):
                    time.sleep(0.25); r=run_predict(model,smi.strip())
                if r: show_result(r)
        else:
            st.markdown(f'<div style="height:360px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:#9AA5BE;gap:.8rem;"><div style="font-size:2.5rem;opacity:.3;">⬡</div><div style="font-family:Poppins;font-size:1rem;">Enter a SMILES string<br>and click {btn_label}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR  — links use ?page=xxx so they actually navigate
# ══════════════════════════════════════════════════════════════════════════════
def nav_class(p): return "nav-btn active" if PAGE==p else "nav-btn"

st.markdown(f"""
<div class="navbar">
  <div class="navbar-brand">
    <div class="navbar-logo">🧬</div>
    <div>
      <div class="navbar-title">Graph<span>AISM</span>Net</div>
      <span class="navbar-subtitle">Anti-Inflammatory Molecular Activity Predictor</span>
    </div>
  </div>
  <div class="navbar-links">
    <a href="?page=home" target="_self"             class="{nav_class('home')}">Home</a>
    <a href="?page=predict_random" target="_self"   class="{nav_class('predict_random')}">Predict (Random)</a>
    <a href="?page=predict_scaffold" target="_self" class="{nav_class('predict_scaffold')}">Predict (Scaffold)</a>
    <a href="?page=batch" target="_self"            class="{nav_class('batch')}">Batch</a>
    <a href="?page=algorithm" target="_self"        class="{nav_class('algorithm')}">Algorithm</a>
    <a href="?page=dataset" target="_self"          class="{nav_class('dataset')}">Dataset</a>
    <a href="?page=help" target="_self"             class="{nav_class('help')}">Help</a>
    <a href="?page=contact" target="_self"          class="{nav_class('contact')}">Contact</a>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "home":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Graph<span>AISM</span>Net</div>
        <div class="hero-sub">A Web Server for Graph Neural Network-Based Anti-Inflammatory Activity Prediction</div>
        <div class="hero-badges">
            <span class="hero-badge green">✓ GCN · GAT · GT · MSMP</span>
            <span class="hero-badge">Random Split (GT) · Scaffold Split (GAT)</span>
            <span class="hero-badge">Classification + Regression</span>
            <span class="hero-badge green">4,300 Compounds · PubChem</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-section">', unsafe_allow_html=True)

    # ── 3-column layout ──────────────────────────────────────────────────────
    col_mol, col_center, col_right = st.columns([1.05, 1.4, 1], gap="medium")

    with col_mol:
        st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:Poppins;font-weight:700;font-size:.95rem;color:#0D2E6E;margin-bottom:.8rem;text-align:center;">Example Molecules</div>', unsafe_allow_html=True)
        sel=st.selectbox("Select compound",list(EXAMPLES.keys()),key="home_mol")
        smi_ex=EXAMPLES[sel]
        svg=mol_to_svg(smi_ex,(380,230))
        if svg:
            b64=base64.b64encode(svg.encode()).decode()
            st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;border-radius:8px;margin:.5rem 0;"/>',unsafe_allow_html=True)
        mol_ex=Chem.MolFromSmiles(smi_ex)
        if mol_ex:
            st.markdown(f"""
            <div style="background:#F7FAFF;border-radius:8px;padding:.7rem;font-size:.82rem;color:#3A4A6B;">
                <div style="font-family:JetBrains Mono;word-break:break-all;color:#1B4FA8;margin-bottom:.4rem;">{smi_ex[:55]}{"…" if len(smi_ex)>55 else ""}</div>
                <div style="display:flex;gap:1.5rem;">
                    <div><b>MW</b> {Descriptors.MolWt(mol_ex):.2f} g/mol</div>
                    <div><b>LogP</b> {Descriptors.MolLogP(mol_ex):.2f}</div>
                </div>
            </div>
            <div style="font-size:.7rem;color:#9AA5BE;margin-top:.4rem;">* Example molecule shown for reference</div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_center:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-bottom:1rem;">
            <div style="font-size:2rem;">🧬</div>
            <div style="font-family:Poppins;font-weight:700;font-size:1.1rem;color:#0D2E6E;margin:.3rem 0;">
                Welcome to GraphAISMNet
            </div>
        </div>
        <div style="font-size:.88rem;line-height:1.8;color:#3A4A6B;text-align:justify;">
        GraphAISMNet is an advanced web-based prediction platform developed using graph neural network
        architectures for accurate molecular activity prediction. The platform integrates graph-based deep
        learning models, including <b>GCN, GAT, GT, and MSMP</b>, to learn intrinsic molecular structural
        and topological information directly from molecular graphs. Built on optimized random split models
        with data augmentation, GraphAISMNet supports both <b>classification and regression predictions</b>,
        enabling reliable identification and quantitative estimation of bioactivity. The platform is designed
        to provide a robust, generalizable, and user-friendly framework for accelerating AI-driven drug
        discovery and molecular screening applications.
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-top:1.2rem;">
            <div style="background:#EEF4FF;border-radius:8px;padding:.7rem;text-align:center;">
                <div style="font-size:1.2rem;">🤖</div>
                <div style="font-family:Poppins;font-weight:600;font-size:.82rem;color:#1B4FA8;">ML-Based</div>
                <div style="font-size:.72rem;color:#6B7A99;">Graph Neural Networks</div>
            </div>
            <div style="background:#E8F8F2;border-radius:8px;padding:.7rem;text-align:center;">
                <div style="font-size:1.2rem;">🎯</div>
                <div style="font-family:Poppins;font-weight:600;font-size:.82rem;color:#0D7A5F;">High Accuracy</div>
                <div style="font-size:.72rem;color:#6B7A99;">AUC up to 0.951</div>
            </div>
            <div style="background:#F0F4FA;border-radius:8px;padding:.7rem;text-align:center;">
                <div style="font-size:1.2rem;">⚡</div>
                <div style="font-family:Poppins;font-weight:600;font-size:.82rem;color:#0D2E6E;">Fast & Reliable</div>
                <div style="font-size:.72rem;color:#6B7A99;">Real-time inference</div>
            </div>
            <div style="background:#FFF8E6;border-radius:8px;padding:.7rem;text-align:center;">
                <div style="font-size:1.2rem;">🔓</div>
                <div style="font-family:Poppins;font-weight:600;font-size:.82rem;color:#B7700A;">Free to Use</div>
                <div style="font-size:.72rem;color:#6B7A99;">Open research tool</div>
            </div>
        </div>
        <div style="background:#FFF8E6;border:1px solid #F5DFA0;border-radius:8px;padding:.7rem 1rem;
             margin-top:1rem;font-size:.8rem;color:#7A5500;text-align:center;">
            ⚠️ For research purposes only. Not for clinical decision making.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card card-green">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.8rem;">
            <span style="font-size:1.3rem;">🚀</span>
            <span style="font-family:Poppins;font-weight:700;font-size:1rem;color:#0D7A5F;">Make a Prediction</span>
        </div>
        <div style="font-size:.85rem;color:#4A5568;margin-bottom:.8rem;">
            Submit a small molecule (SMILES format) to predict its anti-inflammatory activity.
        </div>
        """, unsafe_allow_html=True)
        quick_smi=st.text_input("Enter SMILES string here…",placeholder="e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O",key="home_smi")
        model_choice=st.selectbox("Prediction model",[
            "Random Split — GT (Recommended)","Scaffold Split — GAT"],key="home_model")
        predict_btn=st.button("▶  Predict",key="home_predict")
        if predict_btn and quick_smi.strip():
            mol_chk=Chem.MolFromSmiles(quick_smi.strip())
            if not mol_chk:
                st.markdown('<div class="warn-box">⚠️ Invalid SMILES string.</div>',unsafe_allow_html=True)
            else:
                use_m=gt_model if "Random" in model_choice else gat_model
                with st.spinner("Running inference…"):
                    r=run_predict(use_m,quick_smi.strip())
                if r:
                    css="result-active" if r["active"] else "result-inactive"
                    color="#0D7A5F" if r["active"] else "#C0392B"
                    icon="✅" if r["active"] else "❌"
                    st.markdown(f"""
                    <div class="{css}">
                        <div class="result-label" style="color:{color};">{icon} {"ACTIVE" if r["active"] else "INACTIVE"}</div>
                        <div class="result-sub">Confidence: {r["prob"]*100:.1f}%</div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-top:.6rem;">
                        <div class="stat-pill"><div class="stat-val">{r["pic50"]}</div><div class="stat-lbl">pIC₅₀</div></div>
                        <div class="stat-pill"><div class="stat-val" style="font-size:1rem;">{r["ic50"]}</div><div class="stat-lbl">IC₅₀ (µM)</div></div>
                    </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Help & Documentation
        st.markdown('<div class="card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.8rem;"><span style="font-size:1.2rem;">❓</span><span style="font-family:Poppins;font-weight:700;font-size:1rem;color:#0D2E6E;">Help & Documentation</span></div>', unsafe_allow_html=True)
        for icon,title,sub,link in [
            ("📖","How to Use","Step-by-step guide","?page=help"),
            ("⌨️","Input Format","Accepted SMILES format","?page=help"),
            ("📊","Understanding Results","Interpreting prediction scores","?page=help"),
            ("🔬","Algorithm","Model architecture","?page=algorithm"),
            ("❓","FAQ","Frequently asked questions","?page=help"),
        ]:
            st.markdown(f"""
            <a href="{link}" target="_self" style="text-decoration:none;">
            <div style="display:flex;align-items:center;gap:.7rem;padding:.5rem 0;
                 border-bottom:1px solid #F0F4FA;cursor:pointer;">
                <span style="font-size:1rem;">{icon}</span>
                <div style="flex:1">
                    <div style="font-weight:500;font-size:.85rem;color:#1A2744;">{title}</div>
                    <div style="font-size:.72rem;color:#9AA5BE;">{sub}</div>
                </div>
                <span style="color:#B8CCF0;">›</span>
            </div></a>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end page-section

    # ── Workflow image — full width below 3 cols ───────────────────────────
    if os.path.exists("GRAPH.png"):
        st.markdown("""
        <div style="padding:0 2.5rem 1rem;">
            <div style="background:white;border-radius:14px;padding:1.5rem 2rem;
                 box-shadow:0 2px 16px rgba(13,46,110,.08);border:1px solid rgba(13,46,110,.07);">
                <div style="font-family:Poppins;font-weight:700;font-size:1.1rem;
                     color:#0D2E6E;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem;">
                    📐 GraphAISMNet — Methodology Workflow
                </div>
        """, unsafe_allow_html=True)
        st.image("GRAPH.png", use_container_width=True)
        st.markdown("""
                <div style="font-size:.78rem;color:#9AA5BE;text-align:center;margin-top:.5rem;">
                    Complete methodology pipeline from dataset curation to web deployment
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Performance summary ───────────────────────────────────────────────
    st.markdown('<div style="padding:0 2.5rem 2rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Model Performance Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Best model per split condition — 5-fold cross-validation on test set</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="medium")
    with c1:
        st.markdown("""<div class="card card-blue">
        <div style="font-family:Poppins;font-weight:700;color:#1B4FA8;font-size:.95rem;margin-bottom:.3rem;">🔵 Random Split — GT (Graph Transformer)</div>
        <div style="font-size:.82rem;color:#6B7A99;margin-bottom:.8rem;">Best model for random split with augmentation</div>
        <div class="stat-grid">""", unsafe_allow_html=True)
        for v,l in [("0.950","AUC-ROC"),("89.4%","Accuracy"),("0.789","MCC"),("0.712","R²")]:
            st.markdown(f'<div class="stat-pill"><div class="stat-val">{v}</div><div class="stat-lbl">{l}</div></div>',unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card card-green">
        <div style="font-family:Poppins;font-weight:700;color:#0D7A5F;font-size:.95rem;margin-bottom:.3rem;">🟢 Scaffold Split — GAT (Graph Attention Network)</div>
        <div style="font-size:.82rem;color:#6B7A99;margin-bottom:.8rem;">Best model for scaffold split with augmentation</div>
        <div class="stat-grid">""", unsafe_allow_html=True)
        for v,l in [("0.928","AUC-ROC"),("86.9%","Accuracy"),("0.738","MCC"),("0.621","R²")]:
            st.markdown(f'<div class="stat-pill"><div class="stat-val green">{v}</div><div class="stat-lbl">{l}</div></div>',unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT — RANDOM
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "predict_random":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">🔵 Random Split Prediction</div><div class="hero-sub" style="margin-bottom:.5rem;">Best Model: Graph Transformer (GT) · AUC=0.950 · ACC=89.4%</div><div class="hero-badges"><span class="hero-badge green">✓ Recommended for general use</span><span class="hero-badge">Random stratified split</span><span class="hero-badge">With SMILES augmentation</span></div></div>', unsafe_allow_html=True)
    if not gt_loaded: st.markdown('<div style="padding:1rem 2.5rem 0;"><div class="warn-box">⚠️ model_random_gt.pt not found — running in demo mode.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    predict_panel(gt_model,"GT","▶  Run GT Prediction","r_smi","r_run")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT — SCAFFOLD
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "predict_scaffold":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">🟢 Scaffold Split Prediction</div><div class="hero-sub" style="margin-bottom:.5rem;">Best Model: Graph Attention Network (GAT) · AUC=0.928 · ACC=86.9%</div><div class="hero-badges"><span class="hero-badge">Murcko scaffold split</span><span class="hero-badge green">✓ More rigorous generalisation</span></div></div>', unsafe_allow_html=True)
    if not gat_loaded: st.markdown('<div style="padding:1rem 2.5rem 0;"><div class="warn-box">⚠️ model_scaffold_gat.pt not found — running in demo mode.</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:.5rem 2.5rem 0;"><div class="info-box">ℹ️ Scaffold split tests on structurally novel chemical series. The <a href="?page=predict_random" target="_self" style="color:#1B4FA8;font-weight:600;">Random Split GT model</a> achieves higher accuracy and is recommended for general screening.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    predict_panel(gat_model,"GAT","▶  Run GAT Prediction","s_smi","s_run")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "batch":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📋 Batch Prediction</div><div class="hero-sub">High-throughput screening for multiple molecules</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    bm=st.selectbox("Model for batch prediction",["Random Split — GT (Recommended)","Scaffold Split — GAT"])
    use_bm=gt_model if "Random" in bm else gat_model
    tab1,tab2=st.tabs(["  📁  Upload CSV  ","  ✏️  Paste SMILES  "])
    with tab1:
        st.markdown('<div class="info-box">Upload a CSV with a <b>SMILES</b> column. All other columns are preserved.</div>', unsafe_allow_html=True)
        uploaded=st.file_uploader("Upload CSV",type=["csv"])
        if uploaded:
            df_in=pd.read_csv(uploaded)
            st.write(f"**{len(df_in)}** rows loaded"); st.dataframe(df_in.head(5),use_container_width=True)
            smi_col=next((c for c in df_in.columns if c.lower()=="smiles"),None)
            if not smi_col: st.markdown('<div class="warn-box">⚠️ No SMILES column found</div>',unsafe_allow_html=True)
            else:
                if st.button(f"▶  Predict all {len(df_in)} molecules"):
                    prog=st.progress(0); results=[]
                    for i,smi in enumerate(df_in[smi_col].astype(str)):
                        r=run_predict(use_bm,smi)
                        results.append({"Activity":"Active" if r and r["active"] else "Inactive","Probability":r["prob"] if r else None,"pIC50":r["pic50"] if r else None,"IC50_uM":r["ic50"] if r else None})
                        prog.progress((i+1)/len(df_in))
                    df_out=df_in.copy()
                    for k in ["Activity","Probability","pIC50","IC50_uM"]: df_out[k]=[r[k] for r in results]
                    prog.empty()
                    n_act=sum(1 for r in results if r["Activity"]=="Active")
                    c1b,c2b,c3b=st.columns(3)
                    c1b.metric("Total",len(results)); c2b.metric("Active",n_act); c3b.metric("Inactive",len(results)-n_act)
                    st.dataframe(df_out,use_container_width=True)
                    st.download_button("⬇️ Download Results",df_out.to_csv(index=False),"predictions.csv","text/csv")
    with tab2:
        bulk=st.text_area("One SMILES per line",height=180,placeholder="CC(C)Cc1ccc(cc1)C(C)C(=O)O\nCOc1ccc2cc(C(C)C(=O)O)ccc2c1")
        if st.button("▶  Predict Pasted SMILES"):
            lines=[l.strip() for l in bulk.strip().split("\n") if l.strip()]
            if lines:
                with st.spinner(f"Predicting {len(lines)} molecules…"):
                    res=[{"SMILES":s,"Activity":"Active" if (r:=run_predict(use_bm,s)) and r["active"] else "Inactive","Probability":r["prob"] if r else None,"pIC50":r["pic50"] if r else None,"IC50_uM":r["ic50"] if r else None} for s in lines]
                df_r=pd.DataFrame(res)
                st.dataframe(df_r,use_container_width=True)
                st.download_button("⬇️ Download",df_r.to_csv(index=False),"predictions.csv","text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "algorithm":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📐 Algorithm & Methodology</div><div class="hero-sub">GraphAISMNet — Graph Neural Network Framework</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)

    if os.path.exists("GRAPH.png"):
        st.markdown('<div class="card" style="margin-bottom:2rem;">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:Poppins;font-weight:700;font-size:1rem;color:#0D2E6E;margin-bottom:1rem;">Methodology Workflow</div>', unsafe_allow_html=True)
        st.image("GRAPH.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    for num,title,color,desc in [
        ("1","Dataset Curation","#1B4FA8","4,300 compounds with pIC₅₀ values from PubChem BioAssay. Binary activity: pIC₅₀ ≥ 6.0 → Active. Regression targets standardized (zero mean, unit variance)."),
        ("2","Data Pre-processing","#0D7A5F","Duplicate SMILES removal, missing pIC₅₀ excluded, binary activity labelling, regression target standardization."),
        ("3","Dataset Split","#7B3FA8","Random stratified 80:20 split and Murcko scaffold split. 5-fold cross-validation on both."),
        ("4","Molecular Graph Generation","#1B4FA8","Node features (21-dim): atom type, degree, formal charge, H count, aromaticity, ring, chirality, hybridization. Edge features (7-dim): bond order, conjugation, ring, stereo. Global descriptors (13-dim): MW, LogP, TPSA, QED, HBD/A, Fsp3, ring count, BertzCT, heteroatoms."),
        ("5","Data Augmentation","#0D7A5F","N=3 random SMILES enumerations per molecule, graph edge dropout (15–18%), node feature dropout (5%), test-time augmentation (TTA, 3 variants averaged)."),
        ("6","GNN Models","#7B3FA8","GCN · GAT · GT · MSMP — multi-task learning with Focal Loss + Huber Loss. Class-weighted sampling. Early stopping on composite score (0.6×ACC + 0.4×R²). Ensembling across 2–3 seeds."),
        ("7","Internal Validation","#1B4FA8","5-fold CV per condition. Classification: ACC, AUC, MCC, SN, SP, F1, Precision, BalACC. Regression: R², RMSE, MAE. Statistical significance: paired t-test."),
        ("8","Performance","#0D7A5F","Best models: GT for random split (AUC=0.950, ACC=89.4%), GAT for scaffold split (AUC=0.928, ACC=86.9%). External validation on FDA-approved drugs."),
    ]:
        st.markdown(f"""<div class="card" style="margin-bottom:.8rem;border-left:4px solid {color};">
        <div style="display:flex;gap:1rem;align-items:flex-start;">
            <div style="background:{color};color:white;border-radius:50%;width:28px;height:28px;
                 display:flex;align-items:center;justify-content:center;
                 font-family:Poppins;font-weight:700;font-size:.85rem;flex-shrink:0;">{num}</div>
            <div><div style="font-family:Poppins;font-weight:700;font-size:.92rem;color:#0D2E6E;margin-bottom:.25rem;">{title}</div>
            <div style="font-size:.85rem;color:#4A5568;line-height:1.65;">{desc}</div></div>
        </div></div>""", unsafe_allow_html=True)

    # performance table
    st.markdown('<div style="margin-top:1.5rem;"><div class="section-heading">Performance Tables</div></div>', unsafe_allow_html=True)
    tab_r,tab_s=st.tabs(["Random Split","Scaffold Split"])
    def mk_table(rows, best):
        hdr = "".join("<th>" + h + "</th>" for h in ["Model","AUC","ACC","MCC","SN","SP","F1","R²","RMSE"])
        body = ""
        for r in rows:
            is_best = r[0] == best
            star = " ★" if is_best else ""
            model_cell = '<td class="model-col">' + r[0] + star + "</td>"
            data_cells = ""
            for i, v in enumerate(r[1:]):
                if is_best:
                    data_cells += '<td class="best">' + v + "</td>"
                else:
                    data_cells += "<td>" + v + "</td>"
            body += "<tr>" + model_cell + data_cells + "</tr>"
        return '<div class="card" style="padding:0;overflow:hidden;margin-bottom:1rem;"><table class="perf-table"><thead><tr>' + hdr + '</tr></thead><tbody>' + body + '</tbody></table></div>'
    with tab_r:
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>No Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.941","0.876","0.752","0.893","0.859","0.882","0.687","0.680"],["GAT","0.946","0.884","0.770","0.886","0.883","0.884","0.692","0.673"],["GT","0.943","0.878","0.757","0.880","0.877","0.878","0.680","0.686"],["MSMP","0.946","0.882","0.767","0.912","0.853","0.882","0.682","0.685"]],"GAT"),unsafe_allow_html=True)
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>With Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.944","0.880","0.760","0.903","0.856","0.880","0.698","0.668"],["GAT","0.951","0.892","0.785","0.902","0.882","0.892","0.718","0.644"],["GT","0.950","0.894","0.789","0.909","0.880","0.894","0.712","0.651"],["MSMP","0.951","0.890","0.781","0.888","0.893","0.890","0.709","0.654"]],"GT"),unsafe_allow_html=True)
    with tab_s:
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>No Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.907","0.842","0.686","0.883","0.799","0.842","0.565","0.792"],["GAT","0.919","0.856","0.713","0.880","0.831","0.856","0.594","0.766"],["GT","0.918","0.854","0.709","0.876","0.830","0.854","0.575","0.785"],["MSMP","0.917","0.854","0.709","0.880","0.827","0.854","0.584","0.775"]],"GAT"),unsafe_allow_html=True)
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>With Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.921","0.857","0.713","0.887","0.823","0.857","0.605","0.755"],["GAT","0.928","0.869","0.738","0.885","0.851","0.869","0.621","0.741"],["GT","0.926","0.861","0.724","0.908","0.810","0.861","0.614","0.747"],["MSMP","0.926","0.862","0.724","0.885","0.834","0.862","0.610","0.752"]],"GAT"),unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "dataset":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📂 Dataset</div><div class="hero-sub">Training and evaluation dataset details</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    c1d,c2d=st.columns(2,gap="medium")
    with c1d:
        st.markdown("""<div class="card card-blue">
        <div style="font-family:Poppins;font-weight:700;color:#1B4FA8;font-size:1rem;margin-bottom:.8rem;">Dataset Statistics</div>
        <table class="perf-table" style="font-size:.85rem;">
            <tr><td><b>Source</b></td><td>PubChem BioAssay</td></tr>
            <tr><td><b>Total Compounds</b></td><td>4,300</td></tr>
            <tr><td><b>Active (pIC₅₀ ≥ 6)</b></td><td>2,150 (50%)</td></tr>
            <tr><td><b>Inactive (pIC₅₀ &lt; 6)</b></td><td>2,150 (50%)</td></tr>
            <tr><td><b>pIC₅₀ Mean ± Std</b></td><td>6.45 ± 1.12</td></tr>
            <tr><td><b>Tanimoto Similarity</b></td><td>0.117 ± 0.062</td></tr>
            <tr><td><b>Node Features</b></td><td>21-dim</td></tr>
            <tr><td><b>Edge Features</b></td><td>7-dim</td></tr>
            <tr><td><b>Global Descriptors</b></td><td>13-dim</td></tr>
        </table></div>""", unsafe_allow_html=True)
    with c2d:
        st.markdown("""<div class="card card-green">
        <div style="font-family:Poppins;font-weight:700;color:#0D7A5F;font-size:1rem;margin-bottom:.8rem;">Pre-processing Steps</div>
        <div style="font-size:.87rem;color:#2D3748;line-height:1.9;">
            ✅ Duplicate SMILES removed<br>
            ✅ Missing pIC₅₀ values excluded<br>
            ✅ Binary activity labels (threshold = 6.0)<br>
            ✅ Regression targets standardized<br>
            ✅ Invalid SMILES filtered via RDKit<br>
            ✅ Class balance verified (50:50)<br>
            ✅ Tanimoto similarity analysis performed
        </div>
        <div style="margin-top:1rem;font-family:Poppins;font-weight:700;color:#0D7A5F;font-size:.9rem;">External Validation</div>
        <div style="font-size:.85rem;color:#2D3748;margin-top:.3rem;">FDA-approved drugs used as independent external validation set.</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="margin-top:1rem;">📥 To download the training dataset, please contact the development team.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "help":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">❓ Help & Documentation</div><div class="hero-sub">Guidance on using GraphAISMNet</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    for title,color,content in [
        ("📖 How to Use","#1B4FA8","<b>Step 1:</b> Click <b>Predict (Random)</b> in the navbar (recommended).<br><b>Step 2:</b> Enter a SMILES string or select an example molecule.<br><b>Step 3:</b> Click Run Prediction → get Activity, pIC₅₀, IC₅₀, and properties.<br><b>Step 4:</b> For multiple molecules, use the <b>Batch</b> page."),
        ("⌨️ Input Format","#0D7A5F","Molecules must be in <b>SMILES</b> format.<br><br>Examples:<br>• Ibuprofen: <code>CC(C)Cc1ccc(cc1)C(C)C(=O)O</code><br>• Aspirin: <code>CC(=O)Oc1ccccc1C(=O)O</code><br><br>Batch CSV must have a column named <b>SMILES</b>."),
        ("📊 Understanding Results","#7B3FA8","<b>Activity:</b> Active (pIC₅₀ ≥ 6.0) or Inactive<br><b>Confidence:</b> Sigmoid probability 0–1<br><b>pIC₅₀:</b> Higher = more potent<br><b>IC₅₀ (µM):</b> 10^(6 − pIC₅₀)<br><br><b>Which model?</b> Random Split GT is recommended for general use (higher accuracy)."),
        ("❓ FAQ","#0D2E6E","<b>Q: Is this free?</b> Yes, completely free for research.<br><b>Q: Can I use this clinically?</b> No — research use only.<br><b>Q: Why is scaffold split accuracy lower?</b> It tests on structurally novel compounds not seen during training — a harder, more realistic evaluation.<br><b>Q: Why two models?</b> Random split GT is best overall. Scaffold split GAT provides more conservative estimates for novel chemical series."),
    ]:
        st.markdown(f'<div class="card" style="margin-bottom:1rem;border-left:4px solid {color};"><div style="font-family:Poppins;font-weight:700;font-size:.92rem;color:{color};margin-bottom:.6rem;">{title}</div><div style="font-size:.87rem;color:#2D3748;line-height:1.75;">{content}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONTACT
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "contact":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📬 Contact Us</div><div class="hero-sub">Get in touch with the GraphAISMNet team</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    
    # Use columns to center the contact card (1 part left, 2 parts center, 1 part right)
    _, col_center, _ = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown("""<div class="card card-blue">
        <div style="font-family:Poppins;font-weight:700;color:#1B4FA8;font-size:1rem;margin-bottom:.8rem;">Principal Investigator</div>
        <div style="font-size:.87rem;color:#2D3748;line-height:1.8;">
            <b>Dr. Thirumurthy Madhavan, Ph.D.</b><br>
            Associate Professor<br>
            Principal Investigator — Computational Biology Lab<br>
            Department of Genetic Engineering<br>
            School of Bioengineering<br><br>
            <span style="color:#6B7A99;">SRM Institute of Science and Technology<br>
            Kattankulathur, Kanchipuram District<br>
            Tamil Nadu, India 603203</span><br><br>
            <b>Mobile:</b> +91-9944572918<br>
            <b>Website:</b> <a href="https://www.srmist.edu.in" target="_blank" style="color:#1B4FA8;">www.srmist.edu.in</a>
        </div></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top:1.5rem; text-align: center; font-size:.83rem; color:#4A5568; line-height:1.65;">
            <b style="color:#0D7A5F;">Disclaimer:</b> GraphAISMNet is a computational research tool.
            Predictions require experimental validation.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    © 2025 GraphAISMNet &nbsp;|&nbsp; All rights reserved &nbsp;|&nbsp;
    Developed for research by SRMIST &nbsp;|&nbsp;
    <a href="?page=help" target="_self">Help</a> &nbsp;|&nbsp;
    <a href="?page=contact" target="_self">Contact</a>
</div>
""", unsafe_allow_html=True)