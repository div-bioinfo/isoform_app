
import streamlit as st
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Global Variables ---
transcript_strands = {}  # Initialize the global variable

# --- Utility Functions ---

def sort_exons_by_transcription(exons, strand):
    """
    Sorts exons in transcriptional (5′ to 3′) order based on strand.
    """
    return sorted(exons, key=lambda x: x[0], reverse=(strand == '-'))


def get_5p_3p_coords(exons, strand):
    """
    Returns 5′ and 3′ positions depending on strand.
    For + strand: 5′ is start of first exon, 3′ is end of last exon.
    For - strand: 5′ is end of last exon, 3′ is start of first exon.
    """
    if not exons:
        return None, None
    if strand == '+':
        return exons[0][0], exons[-1][1]
    else:
        return exons[0][1], exons[-1][0]

def parse_gtf_bed(file, file_type, tool_name):
    gene_transcripts = defaultdict(lambda: defaultdict(list))
    try:
        lines = [line.decode('utf-8').strip() for line in file]
    except AttributeError:
        lines = [line.strip() for line in file]

    for line in lines:
        if line.startswith("#") or line == "":
            continue
        fields = line.split('\t')
        if file_type == "GTF":
            if fields[2] != "exon":
                continue
            chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
            attr_str = fields[8]
            gene_id, transcript_id = None, None
            for attr in attr_str.strip().split(';'):
                if 'gene_id' in attr:
                    gene_id = attr.split('"')[1]
                if 'transcript_id' in attr:
                    transcript_id = attr.split('"')[1]
            if gene_id and transcript_id:
                gene_transcripts[gene_id][transcript_id].append((start, end))
                transcript_strands[(tool_name, transcript_id)] = strand
        elif file_type == "BED":
            chrom, start, end, name, score, strand = fields[:6]
            start, end = int(start), int(end)
            blockCount = int(fields[9])
            blockSizes = list(map(int, fields[10].strip(',').split(',')))
            blockStarts = list(map(int, fields[11].strip(',').split(',')))
            gene_id = transcript_id = name
            for i in range(blockCount):
                exon_start = start + blockStarts[i]
                exon_end = exon_start + blockSizes[i]
                gene_transcripts[gene_id][transcript_id].append((exon_start, exon_end))
                transcript_strands[(tool_name, transcript_id)] = strand

    for g in gene_transcripts:
        for t in gene_transcripts[g]:
            strand = transcript_strands.get((tool_name, t), "+")
            gene_transcripts[g][t] = sort_exons_by_transcription(gene_transcripts[g][t], strand)

    return gene_transcripts, transcript_strands

# --- Reference Annotation Loader ---

def parse_reference_gtf(file):
    ref_data = defaultdict(lambda: defaultdict(list))
    ref_strands = {}
    lines = [line.decode('utf-8').strip() if hasattr(line, 'decode') else line.strip() for line in file]
    for line in lines:
        if line.startswith("#") or line == "":
            continue
        fields = line.split('\t')
        if fields[2] != "exon":
            continue
        chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
        attr_str = fields[8]
        gene_id, transcript_id = None, None
        for attr in attr_str.split(';'):
            if 'gene_id' in attr:
                gene_id = attr.split('"')[1]
            if 'transcript_id' in attr:
                transcript_id = attr.split('"')[1]
        if gene_id and transcript_id:
            ref_data[gene_id][transcript_id].append((start, end))
            ref_strands[transcript_id] = strand

    return ref_data, ref_strands

def get_all_gene_ids(data_dict):
    gene_ids = set()
    for tool_dict in data_dict.values():
        gene_ids.update(tool_dict.keys())
    return sorted(gene_ids)

def exon_chains_match(chain1, chain2, strand, tolerance=5):
    chain1 = sort_exons_by_transcription(chain1, strand)
    chain2 = sort_exons_by_transcription(chain2, strand)

    if len(chain1) != len(chain2):
        return False

    for e1, e2 in zip(chain1, chain2):
        if abs(e1[0] - e2[0]) > tolerance or abs(e1[1] - e2[1]) > tolerance:
            return False

    return True

def compute_consensus_novel_with_matches(transcripts_by_tool, transcript_strands_dict, tolerance=5):
    all_chains = []
    for tool, transcripts in transcripts_by_tool.items():
        for tid, exons in transcripts.items():
            strand = transcript_strands_dict.get((tool, tid))
            exons_sorted = sort_exons_by_transcription(exons, strand)
            all_chains.append((tool, tid, exons_sorted, strand))

    exon_status_map = defaultdict(dict)
    consensus_matches = defaultdict(list)

    for tool, transcripts in transcripts_by_tool.items():
        for tid, exons in transcripts.items():
            strand = transcript_strands_dict.get((tool, tid))
            exons_sorted = sort_exons_by_transcription(exons, strand)
            matched_tools = []

            for other_tool, other_tid, other_exons, other_strand in all_chains:
                if tool == other_tool and tid == other_tid:
                    continue
                if strand != other_strand:
                    continue
                if exon_chains_match(exons_sorted, other_exons, strand, tolerance):
                    matched_tools.append(f"{other_tool}:{other_tid}")

            if matched_tools:
                exon_status_map[tool][tid] = "Consensus"
                consensus_matches[f"{tool}:{tid}"] = matched_tools
            else:
                exon_status_map[tool][tid] = "Novel"

    return exon_status_map, consensus_matches

def compute_jaccard_score(exons1, exons2, strand=None):
    if strand:
        exons1 = sort_exons_by_transcription(exons1, strand)
        exons2 = sort_exons_by_transcription(exons2, strand)

    def get_positions(exon_list):
        positions = set()
        for start, end in exon_list:
            positions.update(range(start, end))
        return positions

    pos1 = get_positions(exons1)
    pos2 = get_positions(exons2)
    intersection = pos1.intersection(pos2)
    union = pos1.union(pos2)

    return len(intersection) / len(union) if union else 0.0

def jaccard_similarity(e1, e2):
    """Calculate Jaccard similarity between two exons."""
    start1, end1 = e1
    start2, end2 = e2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union != 0 else 0

# --- Classification Against Reference ---

def classify_transcripts(tool_data, ref_data, ref_strands, tool_name, tolerance=5):
    rows = []
    for gene_id, transcripts in tool_data.items():
        if gene_id not in ref_data:
            for tid, exons in transcripts.items():
                rows.append({
                    "Transcript ID": tid,
                    "Gene ID": gene_id,
                    "Tool": tool_name,
                    "Exons": len(exons),
                    "Classification": "Novel gene",
                    "Structural Tags": "-",
                    "Reference Transcript": "-"
                })
            continue

        ref_transcripts = ref_data[gene_id]

        for tid, exons in transcripts.items():
            best_class = "Novel"
            best_match_tid = "-"
            best_details = "-"
            best_priority = -1
            

            for ref_tid, ref_exons in ref_transcripts.items():
                strand = ref_strands.get(ref_tid, 'NA')
                classification, details = compare_exon_structure_against_reference(
                    ref_exons, exons, strand, tolerance=tolerance
                )

                priority = {
                    "Full match": 3,
                    "Partial match": 2,
                    "Novel combination": 1,
                    "Novel": 0
                }.get(classification, -1)

                if priority > best_priority:
                    best_class = classification
                    best_match_tid = ref_tid
                    best_details = details
                    best_priority = priority

                    if classification == "Full match":
                        break  # No need to look further


            rows.append({
                "Transcript ID": tid,
                "Gene ID": gene_id,
                "Tool": tool_name,
                "Exons": len(exons),
                "Classification": best_class,
                "Structural Tags": best_details,
                "Reference Transcript": best_match_tid if best_match_tid else "-"
            })

    return pd.DataFrame(rows)


def compare_exon_structure(ref_exons, query_exons, strand, tolerance=5, jaccard_threshold=0.7):
    
    """
    Compare exon structures in a strand-aware way and report structure differences.
    Includes 5′/3′ truncation, skipped/extra exons, and boundary mismatches.
    """
    differences = []

    # 1 Sort both exon lists in transcriptional order
    ref_exons = sort_exons_by_transcription(ref_exons, strand)
    query_exons = sort_exons_by_transcription(query_exons, strand)

    # 2 Get strand-aware 5′ and 3′ coordinates
    ref_5p, ref_3p = get_5p_3p_coords(ref_exons, strand)
    query_5p, query_3p = get_5p_3p_coords(query_exons, strand)
    
    # Check 5′ truncation
    if (strand == '+' and query_5p > ref_5p + tolerance) or (strand == '-' and query_5p < ref_5p - tolerance):
        differences.append("5′ truncated")

    # Check 3′ truncation
    if (strand == '+' and query_3p < ref_3p - tolerance) or (strand == '-' and query_3p > ref_3p + tolerance):
        differences.append("3′ truncated")
   

    # 3 Matching exons using Jaccard or containment
    def exons_match(e1, e2):
        return abs(e1[0] - e2[0]) <= tolerance and abs(e1[1] - e2[1]) <= tolerance

    def exon_contains(e_ref, e_query):
        return e_query[0] >= e_ref[0] and e_query[1] <= e_ref[1]

    def jaccard_similarity(e1, e2):
        a = set(range(e1[0], e1[1]))
        b = set(range(e2[0], e2[1]))
        return len(a & b) / len(a | b) if a | b else 0.0

    matched_ref = set()
    matched_query = set() 

    for i, re in enumerate(ref_exons):
        for j, qe in enumerate(query_exons):
            jaccard = jaccard_similarity(re, qe)
            if jaccard >= jaccard_threshold or exon_contains(re, qe):
                matched_ref.add(i)
                matched_query.add(j)
                if not exons_match(re, qe):
                    differences.append(
                        f"Boundary mismatch at exon {i+1} (ref: {re[0]}-{re[1]}) vs (query: {qe[0]}-{qe[1]})"
                    )
                break  # Move to next ref exon after match

    # 4 Detect skipped and extra exons
    for i, re in enumerate(ref_exons):
        if i not in matched_ref:
            differences.append(f"Skipped exon: {re[0]}-{re[1]}")
    for j, qe in enumerate(query_exons):
        if j not in matched_query:
            differences.append(f"Extra exon: {qe[0]}-{qe[1]}")

    return "; ".join(differences) if differences else "Fully matching"

def determine_reference_exons(tool_a, tid_a, exons_a, strand_a,
                              tool_b, tid_b, exons_b, strand_b,
                              reference_selection_mode, manual_reference_tool):
    """
    Returns: (ref_exons, query_exons, ref_tool, ref_tid, ref_iso)
    """
    if reference_selection_mode == "manual" and manual_reference_tool:
        if tool_a == manual_reference_tool:
            return exons_a, exons_b, tool_a, tid_a, "A"
        elif tool_b == manual_reference_tool:
            return exons_b, exons_a, tool_b, tid_b, "B"
    # Fallback or span-based default
    ref_start, ref_end = get_5p_3p_coords(exons_a, strand_a)
    query_start, query_end = get_5p_3p_coords(exons_b, strand_b)
    span_a = abs(ref_end - ref_start)
    span_b = abs(query_end - query_start)
    if span_a >= span_b:
        return exons_a, exons_b, tool_a, tid_a, "A"
    else:
        return exons_b, exons_a, tool_b, tid_b, "B"

def compare_exon_structure_against_reference(ref_exons, query_exons, strand, tolerance=5, jaccard_threshold=0.7):
    
    """
    Compare exon structures in a strand-aware way and report structure differences.
    Includes 5′/3′ truncation, skipped/extra exons, and boundary mismatches.
    """
    differences = []

    # 1 Sort both exon lists in transcriptional order
    ref_exons = sort_exons_by_transcription(ref_exons, strand)
    query_exons = sort_exons_by_transcription(query_exons, strand)

    # 2 Get strand-aware 5′ and 3′ coordinates
    ref_5p, ref_3p = get_5p_3p_coords(ref_exons, strand)
    query_5p, query_3p = get_5p_3p_coords(query_exons, strand)
    
    # Check 5′ truncation
    if (strand == '+' and query_5p > ref_5p + tolerance) or (strand == '-' and query_5p < ref_5p - tolerance):
        differences.append("5′ truncated")

    # Check 3′ truncation
    if (strand == '+' and query_3p < ref_3p - tolerance) or (strand == '-' and query_3p > ref_3p + tolerance):
        differences.append("3′ truncated")
   

    # 3 Matching exons using Jaccard or containment
    def exons_match(e1, e2):
        return abs(e1[0] - e2[0]) <= tolerance and abs(e1[1] - e2[1]) <= tolerance

    def exon_contains(e_ref, e_query):
        return e_query[0] >= e_ref[0] and e_query[1] <= e_ref[1]

    def jaccard_similarity(e1, e2):
        a = set(range(e1[0], e1[1]))
        b = set(range(e2[0], e2[1]))
        return len(a & b) / len(a | b) if a | b else 0.0

    matched_ref = set()
    matched_query = set() 

    for i, re in enumerate(ref_exons):
        for j, qe in enumerate(query_exons):
            jaccard = jaccard_similarity(re, qe)
            if jaccard >= jaccard_threshold or exon_contains(re, qe):
                matched_ref.add(i)
                matched_query.add(j)
                if not exons_match(re, qe):
                    differences.append(
                        f"Boundary mismatch at exon {i+1} (ref: {re[0]}-{re[1]}) vs (query: {qe[0]}-{qe[1]})"
                    )
                break  # Move to next ref exon after match

    # 4 Detect skipped and extra exons
    for i, re in enumerate(ref_exons):
        if i not in matched_ref:
            differences.append(f"Skipped exon: {re[0]}-{re[1]}")
    for j, qe in enumerate(query_exons):
        if j not in matched_query:
            differences.append(f"Extra exon: {qe[0]}-{qe[1]}")

    # Determine category
    if not differences:
        return "Full match", ""
    elif len(matched_ref) > 0 and len(matched_query) > 0:
        return "Partial match", "; ".join(differences)
    elif len(matched_query) == 0:
        return "Novel", "; ".join(differences)
    else:
        return "Novel combination", "; ".join(differences)


def generate_jaccard_table(transcripts_by_tool, transcript_strands_dict, tolerance=5, 
                           reference_selection_mode="span", manual_reference_tool=None):
    rows = []
    tools = list(transcripts_by_tool.keys())

    if reference_selection_mode == "manual" and manual_reference_tool not in transcripts_by_tool:
        st.warning(f"Selected manual reference tool '{manual_reference_tool}' not found in loaded data.")
        return pd.DataFrame()

    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            tool_a, tool_b = tools[i], tools[j]
            transcripts_a = transcripts_by_tool[tool_a]
            transcripts_b = transcripts_by_tool[tool_b]

            for tid_a, exons_a_raw in transcripts_a.items():
                strand_a = transcript_strands_dict.get((tool_a, tid_a))
                exons_a = sort_exons_by_transcription(exons_a_raw, strand_a)

                for tid_b, exons_b_raw in transcripts_b.items():
                    strand_b = transcript_strands_dict.get((tool_b, tid_b))
                    exons_b = sort_exons_by_transcription(exons_b_raw, strand_b)

                    if strand_a != strand_b:
                        continue

                    score = compute_jaccard_score(exons_a, exons_b)

                    # Centralized reference logic
                    ref_exons, query_exons, ref_tool, ref_tid, ref_iso = determine_reference_exons(
                        tool_a, tid_a, exons_a, strand_a,
                        tool_b, tid_b, exons_b, strand_b,
                        reference_selection_mode, manual_reference_tool
                    )

                    ref_strand = transcript_strands_dict.get((ref_tool, ref_tid), strand_a)
                    structure_diff = compare_exon_structure(ref_exons, query_exons, ref_strand, tolerance=tolerance)

                    rows.append({
                        "Transcript A": tid_a,
                        "Tool A": tool_a,
                        "Transcript B": tid_b,
                        "Tool B": tool_b,
                        "Ref_Transcript": ref_iso,
                        "Strand": ref_strand,
                        "Jaccard Score": round(score, 3),
                        "Structure Differences": structure_diff
                    })

    return pd.DataFrame(rows)

def plot_exons_side_by_side(selected_transcripts):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, len(selected_transcripts) * 1.5))
    y_pos = range(len(selected_transcripts), 0, -1)

    for i, (tool, tid, exons) in enumerate(selected_transcripts):
        color = "#EC407A" if tool == "Reference" else "tab:blue"  # 🎨 Pink for reference
        label = "Reference" if tool == "Reference" else tool
        for exon in exons:
            ax.broken_barh(
                [(exon[0], exon[1] - exon[0])],
                (y_pos[i] - 0.4, 0.8),
                facecolors=color
            )
        ax.text(
            min([e[0] for e in exons]) - 50,
            y_pos[i],
            f"{label}:{tid}",
            verticalalignment="center",
            fontsize=9
        )

    ax.set_xlabel("Genomic position")
    ax.set_yticks([])
    ax.set_title("Isoform Exon Structures ", fontsize=13, fontweight="bold")
    st.pyplot(fig)





# --- Streamlit UI with Enhanced Styling and Background ---

st.set_page_config(page_title="Isoform Comparison Dashboard", layout="wide")
st.markdown(
    """
    <style>
    /* Fix layout shift when sidebar opens/closes */
    .stApp {
        padding-left: 2rem;
        padding-right: 2rem;
        transition: padding 0.3s ease;
    }

    /* Optional: limit main content width and center it */
    section.main > div {
        max-width: 1200px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Inject custom CSS and background image (you can replace the URL with any suitable image)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;800&display=swap');

    /* Background with subtle gradient overlay */
    .stApp {
        background: linear-gradient(
            rgba(255,255,255,0.4),
            rgba(255,255,255,0.4)
            ),
            url('https://github.com/div-bioinfo/isoform_app/blob/main/background.jpg?raw=true') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Poppins', sans-serif;
        color: #2C2C2C; /*deep grey for general text*/
    }
    .main-header {
        font-family: 'Poppins', sans-serif !important;
        color: #7C1C5A !important; /* rich plum  */
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        margin-bottom: 0 !important;
        text-shadow: 1px 1px 3px #fcd34d !important;
    }

    .sub-header {
        font-family: 'Poppins', sans-serif !important;
        color: #5D1451 !important; /* orchid violet */
        font-size: 1.25rem !important;
        margin-top: 0 !important;
        margin-bottom: 1.5rem !important;
        font-style: italic !important;
    }

    /* Section titles */
    h2, .section-title {
        color: #3F3C6E !important;  /* muted indigo */
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Dropdown text and labels */
    label, .stSelectbox label {
        font-weight: 600;
        color: #46344E !important;  /* deep lavender gray */
    }

    /* Dropdown selected text */
    .css-1cpxqw2, .stSelectbox div[data-baseweb="select"] {
        color: #2C2C2C !important;  /* strong readable dark gray */
    }


    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #fff9e6;  /* soft warm cream */
        color: #4b5563;
        padding: 2rem 1rem;
        font-size: 1rem;
        border-right: 1px solid #fcd34d;
    }

    [data-testid="stSidebar"] h2 {
        color: #f59e0b;  /* bright amber */
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px #fde68a;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2563eb;  /* bright blue */
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
        font-size: 1rem;
        transition: background-color 0.3s ease;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1e40af;
        cursor: pointer;
    }

    /* Dataframe and editor container */
   .stDataFrame, .stDataEditor {
       border-radius: 12px;
       box-shadow: 0 6px 12px rgb(173 216 230 / 0.15); /* soft baby blue glow */
       background-color: #f9fbfc; /* off-white, slightly cool */
       padding: 1rem;
       font-family: 'Poppins', sans-serif;
   }

   /* Table headers */
   .stDataFrame thead tr th {
       background-color: #5a9bd5 !important; /* medium baby blue */
       color: white !important;
       font-weight: 700 !important;
       text-align: center !important;
       padding: 0.6rem 1rem !important;
       border-bottom: 3px solid #4178be !important; /* deeper blue border */
   }

   /* Table body rows */
   .stDataFrame tbody tr:nth-child(even) {
       background-color: #e3f2fd !important; /* very light baby blue */
   }

   .stDataFrame tbody tr:nth-child(odd) {
       background-color: #f9fbfc !important; /* off-white */
   }

   /* Table cells */
   .stDataFrame tbody tr td {
       padding: 0.6rem 1rem !important;
       color: #2c3e50 !important; /* dark slate blue */
       text-align: center !important;
       border-bottom: 1px solid #bbdefb !important;

    }

    /* Radio buttons horizontal alignment */
    div[role="radiogroup"] > label {
        margin-right: 2rem;
        font-weight: 600;
        color: #1f2937;
    }

    /* File uploader */
    div.stFileUploader {
        background: #e0e7ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
    }

    /* About section styling for main page */
    .about-box {
       background-color: rgba(255, 247, 230, 0.88);
       padding: 1.2rem;
       border-radius: 12px;
       font-size: 0.87rem;
       line-height: 1.6;
       box-shadow: 0 3px 8px rgba(0,0,0,0.06);
       margin-top: 1.2rem;
    }
    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: #7C1C5A;    /* rich plum  */
        margin-bottom: 0.8rem;
    }
   
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title and description
st.markdown('<h1 class="main-header">🧬 Isoform Comparison Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Compare isoform structures across tools or against reference.</p>',
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:    
    st.markdown("### 🔍 Select Comparison Mode")
    mode = st.radio("Choose mode", ["Compare across Tools", "Compare against a Reference"], horizontal=False, key='comparision_mode')
    reference_selection_mode = None
    manual_reference_tool = None
    
    st.markdown("---")
    
    # Main tool files upload (always shown)
    uploaded_files = st.file_uploader(
        "📁 Upload Tool Output Files (GTF/BED)",
        type=["gtf"],
        accept_multiple_files=True,
        help="""Upload isoform prediction outputs from different tools/sources. Upload limit 4.
        📝 **File Naming Tip**: (Trust Us, It Helps!) Name your files like:
        - `sample_flair.gtf`
        - `experiment_GENECODE.gtf`
        If the file name **doesn't contain** one of: `flair`, `talon`, `isoquant`, or `stringtie2`,
        the app will treat the tool name as the file name itself, so make sure the sourse name for file is present
        — else it might make your comparisons confusing!"""
    )

    ref_file = None
    ref_filename = ""

    # Mode-specific controls
    if mode == "Compare across Tools":
        
        st.markdown("---")
        reference_selection_mode = st.radio(
            "📌 Reference Transcript Selection Mode",
            ["Span-based", "Manual (choose tool)"],
            help="Span-based uses longer transcript as reference. Manual allows you to pick the reference tool."
        )
    
        if reference_selection_mode == "Manual (choose tool)":
            reference_selection_flag = "manual"  # Set the mode flag
        
            # Only proceed if files have been uploaded
            if uploaded_files:
                tool_name_map = {}
                display_options = []

                for uploaded_file in uploaded_files:
                    name = uploaded_file.name.lower()
                    tool = (
                        "FLAIR" if "flair" in name else
                        "TALON" if "talon" in name else
                        "Isoquant" if "isoquant" in name else
                        "Stringtie2" if "stringtie2" in name else
                        uploaded_file.name.split('.')[0]
                    )
                    display_name = uploaded_file.name.split('.')[0]
                    tool_name_map[display_name] = tool
                    display_options.append(display_name)

                selected_display_name = st.selectbox(
                    "🛠️ Choose Reference Tool",
                    options=display_options,
                    help="All transcripts from this tool will be treated as the reference in comparisons."
                )
                manual_reference_tool = tool_name_map[selected_display_name]
            else:
                st.warning("Please upload tool files first to select a reference tool")
                manual_reference_tool = None

    elif mode == "Compare against a Reference":
            
        st.markdown("---")
        st.markdown("### 🧬 Reference Annotation")
        ref_file = st.file_uploader(
            "📁 Upload Reference Annotation (GTF)", 
            type=["gtf"],
            help="Upload reference annotation from GENCODE/Ensembl or similar sources",
            key='reference_file'
        )
        if ref_file:
            # Only show overlap slider if reference is uploaded
            min_exon_overlap = st.slider(
                "Minimum exon overlap ratio", 
                min_value=0.1, max_value=1.0, value=0.7, step=0.1
            )
    st.markdown("---")        
    st.markdown("### ⚙️ Comparison Settings")
    tolerance = st.slider(
        "🧬 Exon Boundary Tolerance (in bp)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Controls how strict the matching should be between exon boundaries across isoforms."
    )
           
    

# Main content area
if not uploaded_files:
    st.markdown("""
<div class='about-box'>
<div class='section-heading'>🧭 About This App</div>
 
Welcome to the **Isoform Comparison Dashboard** — your friendly companion for exploring, comparing, and decoding isoforms predicted by different long-read transcriptome analysis tools.

This app lets you:
- 🔍 Visualize isoform structures and compare across tools
- 📊 Identify **novel** and **consensus** isoforms at a glance

---

<div class='section-heading'>👩‍💻 Author</div>

Built with  🧠,🐍,and a little caffeine by **Divya**,
an M.Sc. Bioinformatics student<br>
Bioinformatics Centre, SPPU<br>
GeneSpectrum Life Sciences LLP.
</div>
""", unsafe_allow_html=True)


if uploaded_files:
    data_by_tool = {}
    transcript_strands_dict = {}

    # Identify reference file name to exclude it from tool GTF parsing
    ref_filename = ref_file.name if ref_file else ""


    for uploaded_file in uploaded_files:
        if uploaded_file.name == ref_filename:
            continue  # Skip parsing reference here
        name = uploaded_file.name.lower()
        tool = "FLAIR" if "flair" in name else "TALON" if "talon" in name else "Isoquant" if "isoquant" in name else 'Stringtie2' if "stringtie2" in name else uploaded_file.name.split('.')[0]
        file_type = "GTF" if name.endswith(".gtf") else "BED"
        gene_transcripts, updated_strands = parse_gtf_bed(uploaded_file, file_type, tool)
        data_by_tool[tool] = gene_transcripts
        transcript_strands_dict.update(updated_strands)

    # Load and store reference data separately
    if mode == "Compare against a Reference" and ref_file:
        ref_data, ref_strands = parse_reference_gtf(ref_file)
        transcript_strands_dict.update(ref_strands)

    # This runs in BOTH modes — get all tool-based gene IDs
    all_genes = get_all_gene_ids(data_by_tool)
    gene_id = st.selectbox("### 🧬 Select Gene ID", all_genes)

    if gene_id:
        transcripts_by_tool = {
            tool: data_by_tool[tool][gene_id]
            for tool in data_by_tool if gene_id in data_by_tool[tool]
        }

        if mode == "Compare against a Reference" and ref_file:
            
            # Run classification per tool and store in a dictionary
            classified_tables = {}

            for tool in transcripts_by_tool:
                tool_data = {gene_id: transcripts_by_tool[tool]}
                df = classify_transcripts(
                    tool_data,
                    ref_data,
                    ref_strands,
                    tool_name=tool,
                    tolerance=tolerance
                )

                # Add dynamic columns
                df["Exons"] = df["Transcript ID"].apply(lambda tid: len(transcripts_by_tool[tool].get(tid, [])))
                df["Strand"] = df["Transcript ID"].apply(lambda tid: transcript_strands_dict.get((tool, tid), "?"))
                df["Select for Plot"] = False  # This allows interactive selection

                # Rename for UI consistency
                df = df.rename(columns={
                    "Classification": "Isoform Status",
                    
                })

                # Store this tool's table
                classified_tables[tool] = df[df["Gene ID"] == gene_id]  # Only include selected gene

            # --- 📊 Stacked Bar Plot of Isoform Status ---
            st.markdown("### 📊 Isoform Classification Summary")

            # 1. Gather status counts from all per-tool tables
            plot_counts = defaultdict(lambda: defaultdict(int))
            for tool, df in classified_tables.items():
                for status, count in df["Isoform Status"].value_counts().items():
                    plot_counts[tool][status] += count

            # 2. Convert to DataFrame
            plot_data = []
            for tool, status_counts in plot_counts.items():
                for status, count in status_counts.items():
                    plot_data.append({"Tool": tool, "Isoform Status": status, "Count": count})
            df_plot = pd.DataFrame(plot_data)

            # 3. Pivot and normalize
            pivot_df = df_plot.pivot(index="Tool", columns="Isoform Status", values="Count").fillna(0)
            pivot_percent = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

            # 4. Define consistent color map
            colors = {
                "Full match": "#4CAF50",
                "Partial match": "#8BC34A",
                "Novel combination": "#2196F3",
                "Novel": "#F44336",
                "Novel gene": "#9C27B0"
            }

            # 5. Plot
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#f7f7f7')
            ax.set_facecolor('#f7f7f7')

            bottom = np.zeros(len(pivot_percent))
            x_positions = np.arange(len(pivot_percent))

            for status in pivot_percent.columns:
                if status in colors:
                    values = pivot_percent[status].values
                    bars = ax.bar(
                        pivot_percent.index, values, bottom=bottom,
                        label=status, color=colors[status], width=0.4,
                        edgecolor='white', linewidth=0.5
                    )
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_y() + height / 2,
                                f"{height:.0f}%",
                                ha="center", va="center", fontsize=9, color="black"
                            )
                    bottom += values

            # 6. Final touches
            ax.set_ylabel("Percentage", fontsize=10)
            ax.set_title("Isoform Classification Against Reference", fontsize=12, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(pivot_percent.index, fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(title="Isoform Status", title_fontsize=9, fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
            fig.tight_layout()

            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
            

                st.pyplot(fig)

            # --- Display per-tool tables in tabs ---
            st.markdown("### 📋 Isoform Classification Summary (by Tool)")
            st.markdown(
                f"""
                <div style='background-color:#e3f2fd;padding:6px 10px;border-radius:8px;'>
                    <p style='margin:0;color:#4A148C;font-size:14px;'>📖 Reference used: <b>{ref_file.name}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
                
                
            tab_list = st.tabs(list(classified_tables.keys()))
            for i, tool in enumerate(classified_tables.keys()):
                with tab_list[i]:
                    st.markdown(f"🧪 Tool: {tool}")
                   
                    df_summary = classified_tables[tool]

                    # Reorder columns as requested
                    desired_order = [
                        "Transcript ID",
                        "Exons",
                        "Reference Transcript",
                        "Isoform Status",
                        "Structural Tags",
                        "Strand",
                        "Select for Plot"
                    ]
                    df_summary = df_summary.reindex(columns=desired_order)

                    edited_df = st.data_editor(
                        df_summary,
                        use_container_width=True,
                        num_rows="dynamic",
                        key=f"{tool}_summary_editor"
                    )

                    # --- Visualize selected isoforms ---
                    selected_df = edited_df[edited_df["Select for Plot"] == True]
                    if not selected_df.empty:
                        st.markdown("### 🧬 Visualize Selected Isoforms with Reference")
                        selected_transcripts = []
                        for _, row in selected_df.iterrows():
                            tid = row["Transcript ID"]
                            matched_tid = row["Reference Transcript"]
                            try:
                                 exons = transcripts_by_tool[tool][tid]
                                 selected_transcripts.append((tool, tid, exons))

                                 if matched_tid != "-" and matched_tid in ref_data.get(gene_id, {}):
                                     ref_exons = ref_data[gene_id][matched_tid]
                                     selected_transcripts.append(("Reference", matched_tid, ref_exons))
                            except KeyError:
                                st.warning(f"Transcript {tid} from {tool} not found.")
                        if selected_transcripts:
                            plot_exons_side_by_side(selected_transcripts)
                    else:
                        st.info("✅ Select isoforms using the checkboxes in the table above to auto-plot them.")

        else:
            exon_status, consensus_matches = compute_consensus_novel_with_matches(
                transcripts_by_tool, transcript_strands_dict, tolerance=tolerance
            )

            rows = []
            seen_transcripts = set()

            for tool, transcripts in transcripts_by_tool.items():
                for tid, exons in transcripts.items():
                    key = f"{tool}:{tid}"
                    if key in seen_transcripts:
                        continue

                    strand = transcript_strands_dict.get((tool, tid), "?")
                    status = exon_status[tool].get(tid, "Unknown")

                    if status == "Consensus":
                        matched_list = consensus_matches.get(key, [])
                        matched_with = " | ".join(matched_list) if matched_list else "–"
                        seen_transcripts.add(key)
                        seen_transcripts.update(matched_list)
                    else:
                        matched_with = "–"
                        seen_transcripts.add(key)

                    rows.append({
                        "Tool": tool,
                        "Transcript ID": tid,
                        "Exons": len(exons),
                        "Strand": strand,
                        "Isoform Status": status,
                        "Matched With": matched_with,
                        "Select for Plot": False,
                    })

            df_summary = pd.DataFrame(rows)

            # Count all transcript status per tool (including matched transcripts)
            plot_counts = defaultdict(lambda: defaultdict(int))

            # 1. Count visible transcripts
            for _, row in df_summary.iterrows():
                tool = row["Tool"]
                status = row["Isoform Status"]
                plot_counts[tool][status] += 1

            # 2. Include additional consensus-matched transcripts
            for key, matched_list in consensus_matches.items():
                tool_a, tid_a = key.split(":", 1)
                for match_key in matched_list:
                    tool_b, tid_b = match_key.split(":", 1)
                    if not ((df_summary["Tool"] == tool_b) & (df_summary["Transcript ID"] == tid_b)).any():
                        # If transcript not already in summary, count it
                        plot_counts[tool_b]["Consensus"] += 1

            # Convert to DataFrame
            expanded_rows = []
            for tool, status_counts in plot_counts.items():
                for status, count in status_counts.items():
                    expanded_rows.append({"Tool": tool, "Isoform Status": status, "Count": count})

            df_plot = pd.DataFrame(expanded_rows)


            st.markdown("### 📊 Graphic Isoform Summary")
            # Pivot for stacking
            summary_pivot = df_plot.pivot(index="Tool", columns="Isoform Status", values="Count").fillna(0)
            summary_pivot_percent = summary_pivot.div(summary_pivot.sum(axis=1), axis=0) * 100
            
            # Plotting
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#f7f7f7')
            ax.set_facecolor('#f7f7f7')
            bottom = None
            if mode == "Compare against a Reference" and ref_file:
                colors = {
                    "Full match": "#2E7D32",
                    "Partial match": "#8BC34A",
                    "Novel combination": "#2196F3",
                    "Novel": "#F44336",
                    "Novel gene": "#9C27B0"
                }
            else:
                colors = {"Consensus": "#A78BFA", "Novel": "#FCD34D"}  # lavender & amber
            
            bar_width = 0.4
            x_positions = np.arange(len(summary_pivot_percent))
            bottom = np.zeros(len(summary_pivot_percent))


            for status in summary_pivot_percent.columns:
                if status in colors:
                    values = summary_pivot_percent[status]
                    bars = ax.bar(summary_pivot_percent.index, values, bottom=bottom, 
                                 label=status, color=colors[status], width=bar_width, 
                                 edgecolor='white', linewidth=0.5)

            

                    # Label inside each segment
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_y() + height / 2,
                                f"{height:.0f}%",
                                ha="center",
                                va="center",
                                fontsize=9,
                                color="black"
                                )
                    bottom += values
                    
            ax.set_xlim(-0.5, len(x_positions) - 0.5)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percentage", fontsize=10)
            title = "Isoform Classification Against Reference" if (mode == "Compare against a Reference" and ref_file) else "Consensus vs Novel Isoforms per Tool"
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylim(0,100)
            
            #remove top and right borders
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
                
            ax.legend(title="Isoform Status", title_fontsize=10, fontsize=9,frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))

            # X-axis with tool names
            ax.set_xticks(x_positions)
            ax.set_xticklabels(summary_pivot_percent.index, fontsize=10)

            # Tight layout to prevent overlap
            fig.tight_layout()
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
   
                st.pyplot(fig)              


            # --- Summary Table Section ---
            st.markdown("---")
            st.markdown("### 📋 Isoform Summary Table")
            st.markdown("Select one or more transcripts below to view their exon structure.")

            # Show editable summary table
            edited_df = st.data_editor(
                df_summary,
                use_container_width=True,
                num_rows="dynamic",
                key="summary_editor"
            )

            # Extract selected transcripts
            selected_df = edited_df[edited_df["Select for Plot"] == True]

            # --- Auto Visualization ---
            if not selected_df.empty:
                st.markdown("### 🧬 Visualize Selected Isoforms")
                selected_transcripts = []
                for _, row in selected_df.iterrows():
                    tool = row["Tool"]
                    tid = row["Transcript ID"]
                    try:
                        exons = transcripts_by_tool[tool][tid]
                        selected_transcripts.append((tool, tid, exons))
                    except KeyError:
                        st.warning(f"Transcript {tid} from {tool} not found.")
                if selected_transcripts:
                    plot_exons_side_by_side(selected_transcripts)
            else:
                st.info("✅ Select isoforms using the checkboxes in the table above to auto-plot them.")


            st.markdown("---")
            st.markdown("### 🔗 Pairwise Structure Comparison")
            jaccard_df = generate_jaccard_table(
                transcripts_by_tool,
                transcript_strands_dict,
                tolerance=tolerance,
                reference_selection_mode=reference_selection_mode,
                manual_reference_tool=manual_reference_tool
            )
            if jaccard_df is not None and not jaccard_df.empty:
                available_tools = sorted(set(jaccard_df["Tool A"]).union(set(jaccard_df["Tool B"])))
                #Tool pair selection
                st.markdown("###🎯 Select File Pair for Filtering")
                col1, col2 = st.columns(2)

                with col1:
                    selected_tool_a = st.selectbox("🔧 File/Tool A", available_tools, key="tool_a_select")

                with col2:
                    selected_tool_b = st.selectbox("🔩 File/Tool B", [t for t in available_tools if t != selected_tool_a], key="tool_b_select")

                show_all = st.checkbox("📋 Show All Tool Comparisons", value=False)

                #Filter jaccard dataframe based on selected tools
                if not show_all:
                    filtered_jaccard_df = jaccard_df[
                        ((jaccard_df["Tool A"] == selected_tool_a) & (jaccard_df["Tool B"] == selected_tool_b)) |
                        ((jaccard_df["Tool A"] == selected_tool_b) & (jaccard_df["Tool B"] == selected_tool_a))
                    ]

                    st.markdown(f"###🧪 Comparison: `{selected_tool_a}` vs `{selected_tool_b}`")
                    st.dataframe(filtered_jaccard_df, use_container_width=True)

                else:
                    st.markdown("### 🗂️ All Pairwise Comparisons")
                    st.dataframe(jaccard_df, use_container_width=True)
                
        
            
else:
    st.info("""
👆 Upload GTF files to begin analysis.

📝 **File Naming Tip**: (Trust Us, It Helps!)
Name your files like:
- `sample_flair.gtf`
- `experiment_GENECODE.gtf`

If the file name **doesn't contain** one of: `flair`, `talon`, `isoquant`, or `stringtie2`, 
the app will treat the tool name as the file name itself — which might make your comparisons confusing!
""")

   
             
