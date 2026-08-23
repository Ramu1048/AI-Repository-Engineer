# app.py
import streamlit as st
import requests
import json
import time
import os

# Configure page metadata and premium styling
st.set_page_config(
    page_title="AI Repository Engineer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Titles and Headings */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #3F3D56 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metrics Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-bottom: 15px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #88888d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6C63FF;
        margin-top: 5px;
    }
    
    /* Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-error {
        background-color: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    
    .badge-warning {
        background-color: rgba(255, 165, 0, 0.15);
        color: #FFA500;
        border: 1px solid rgba(255, 165, 0, 0.3);
    }
    
    .badge-info {
        background-color: rgba(0, 191, 255, 0.15);
        color: #00BFFF;
        border: 1px solid rgba(0, 191, 255, 0.3);
    }
    
    .badge-success {
        background-color: rgba(0, 200, 83, 0.15);
        color: #00C853;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    
    /* Code container styling */
    .source-block {
        border-left: 4px solid #6C63FF;
        background-color: rgba(108, 99, 255, 0.05);
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Base URL
backend_url = "http://localhost:8000"
try:
    if "BACKEND_URL" in st.secrets:
        backend_url = st.secrets["BACKEND_URL"]
    elif os.getenv("BACKEND_URL"):
        backend_url = os.getenv("BACKEND_URL")
except Exception:
    if os.getenv("BACKEND_URL"):
        backend_url = os.getenv("BACKEND_URL")

API_URL = backend_url.rstrip("/") + "/api"

# Initialize Session State
if "repository_id" not in st.session_state:
    st.session_state.repository_id = None
if "repo_status" not in st.session_state:
    st.session_state.repo_status = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prev_url" not in st.session_state:
    st.session_state.prev_url = ""

# Sidebar controls
st.sidebar.title("🤖 Config Panel")
github_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")
llm_provider = st.sidebar.selectbox("LLM Provider", ["gemini", "ollama"], index=0)
embedding_provider = st.sidebar.selectbox("Embedding Provider", ["gemini", "ollama"], index=0)
top_k = st.sidebar.slider("Top-K Retrieve Chunks", min_value=1, max_value=25, value=8)

st.sidebar.caption(f"Backend Server: `{backend_url}`")
st.sidebar.markdown("---")

# Detect repository URL switch to clear old state (Rule 5)
if github_url != st.session_state.prev_url:
    st.session_state.repository_id = None
    st.session_state.repo_status = None
    st.session_state.chat_history = []
    st.session_state.prev_url = github_url

# Analyze action button
analyze_button = st.sidebar.button("🔍 Analyze Repository", use_container_width=True)

if analyze_button:
    if not github_url:
        st.sidebar.error("Please enter a valid GitHub URL first.")
    else:
        with st.spinner("Analyzing repository... Ingesting, chunking and indexing."):
            try:
                # Trigger ingestion API
                response = requests.post(
                    f"{API_URL}/analyze",
                    json={
                        "github_url": github_url,
                        "llm_provider": llm_provider,
                        "top_k": top_k
                    },
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.repo_status = data
                    st.session_state.repository_id = data["repository_id"]
                    st.sidebar.success("Repository analysis complete!")
                else:
                    err_detail = response.json().get("detail", "Unknown server error")
                    st.sidebar.error(f"Analysis failed: {err_detail}")
            except Exception as e:
                st.sidebar.error(f"Error connecting to backend: {e}")

# Header
st.title("🤖 AI Repository Engineer")
st.markdown("---")

if not st.session_state.repository_id:
    st.warning("👈 Please enter a GitHub Repository URL and click **Analyze Repository** in the sidebar to begin.")
else:
    repo_id = st.session_state.repository_id
    status_data = st.session_state.repo_status or {}
    owner = status_data.get("owner")
    name = status_data.get("name")
    
    # Fallback to parse from repository_id
    if not owner or not name:
        if "/" in repo_id:
            owner, name = repo_id.split("/", 1)
        elif "_" in repo_id:
            parts = repo_id.split("_", 2)
            owner = parts[0]
            name = parts[1] if len(parts) > 1 else repo_id
        else:
            owner, name = "unknown", repo_id
            
    # Refresh status
    try:
        status_resp = requests.get(f"{API_URL}/repository/{owner}/{name}")
        if status_resp.status_code == 200:
            st.session_state.repo_status = status_resp.json()
    except Exception:
        pass
        
    status_data = st.session_state.repo_status
    
    # Main Tabs
    tab_chat, tab_overview, tab_arch, tab_deps, tab_bugs, tab_sec, tab_issues, tab_prs = st.tabs([
        "💬 Chat",
        "📋 Repository Overview",
        "🏗️ Architecture",
        "📦 Dependencies",
        "🪲 Bug Analysis",
        "🔒 Security",
        "🎫 GitHub Issues",
        "🔌 Pull Requests"
    ])
    
    # ----------------- Tab 1: Chat -----------------
    with tab_chat:
        st.subheader("Chat with your Codebase")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("Show Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"📁 **{src['file_path']}** (lines {src['start_line']} - {src['end_line']})")
                            
        # Chat input
        user_question = st.chat_input("Ask a question about the code...")
        if user_question:
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            with st.spinner("Generating grounded answer..."):
                try:
                    chat_resp = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "repository_id": repo_id,
                            "question": user_question,
                            "top_k": top_k
                        }
                    )
                    if chat_resp.status_code == 200:
                        ans_data = chat_resp.json()
                        with st.chat_message("assistant"):
                            st.write(ans_data["answer"])
                            if ans_data["sources"]:
                                with st.expander("Show Sources"):
                                    for src in ans_data["sources"]:
                                        st.markdown(f"📁 **{src['file_path']}** (lines {src['start_line']} - {src['end_line']})")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": ans_data["answer"],
                            "sources": ans_data["sources"]
                        })
                    else:
                        err = chat_resp.json().get("detail", "Error generating response")
                        st.error(err)
                except Exception as e:
                    st.error(f"Failed to communicate with API: {e}")

    # ----------------- Tab 2: Repository Overview -----------------
    with tab_overview:
        st.subheader("Repository Overview")
        if status_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Repository Name</div>
                    <div class="metric-value">{status_data['name']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Owner</div>
                    <div class="metric-value">{status_data['owner']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Indexed Files</div>
                    <div class="metric-value">{status_data['file_count']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Chunks count</div>
                    <div class="metric-value">{status_data['chunk_count']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"**Description:** {status_data['description'] or 'No description'}")
            st.markdown(f"**Primary Language:** `{status_data['primary_language']}`")
            st.markdown(f"**All Detected Languages:** {', '.join(status_data['languages'])}")
            
            # Status Badge
            status_map = {
                "ready": ("badge-success", "READY"),
                "indexing": ("badge-warning", "INDEXING"),
                "failed": ("badge-error", "FAILED"),
                "not_started": ("badge-info", "NOT STARTED")
            }
            bg_class, text = status_map.get(status_data['indexing_status'], ("badge-info", "UNKNOWN"))
            st.markdown(f"**Indexing Status:** <span class='badge {bg_class}'>{text}</span>", unsafe_allow_html=True)

    # ----------------- Tab 3: Architecture -----------------
    with tab_arch:
        st.subheader("Architecture Dependency Graph")
        with st.spinner("Fetching architecture report..."):
            try:
                arch_resp = requests.get(f"{API_URL}/repository/{owner}/{name}/architecture")
                if arch_resp.status_code == 200:
                    arch_data = arch_resp.json()
                    st.markdown("### Architectural Summary")
                    st.write(arch_data["summary"])
                    
                    st.markdown("### Mermaid Diagram")
                    # Render Mermaid JS inside Streamlit
                    mermaid_code = arch_data["mermaid_diagram"]
                    st.code(mermaid_code, language="mermaid")
                    
                    st.markdown("#### Rendered Flowchart")
                    html_code = f"""
                    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                    <script>mermaid.initialize({{startOnLoad:true}});</script>
                    <div class="mermaid">
                    {mermaid_code}
                    </div>
                    """
                    st.components.v1.html(html_code, height=300, scrolling=True)
                else:
                    st.error("Failed to load architecture report.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ----------------- Tab 4: Dependencies -----------------
    with tab_deps:
        st.subheader("Discovered Dependencies")
        with st.spinner("Loading dependencies..."):
            try:
                deps_resp = requests.get(f"{API_URL}/repository/{owner}/{name}/dependencies")
                if deps_resp.status_code == 200:
                    deps_list = deps_resp.json()
                    if deps_list:
                        st.dataframe(deps_list, use_container_width=True)
                    else:
                        st.info("No dependency files (like package.json, requirements.txt) detected.")
                else:
                    st.error("Failed to load dependencies.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ----------------- Tab 5: Bug Analysis -----------------
    with tab_bugs:
        st.subheader("Static Code Quality & Bug Findings")
        with st.spinner("Analyzing code for potential bugs..."):
            try:
                bugs_resp = requests.get(f"{API_URL}/repository/{owner}/{name}/bugs")
                if bugs_resp.status_code == 200:
                    bugs_list = bugs_resp.json()
                    if not bugs_list:
                        st.success("No code issues or bugs identified by static analysis! Nice job.")
                    else:
                        for finding in bugs_list:
                            sev_class = "badge-error" if finding['severity'] == "error" else "badge-warning"
                            st.markdown(f"""
                            ### 📁 `{finding['file_path']}` (Line {finding['line']})
                            **Severity:** <span class="badge {sev_class}">{finding['severity'].upper()}</span> | **Confidence:** `{finding['confidence']}` | **Tool:** `{finding['tool']}`
                            
                            **Issue Details:**  
                            *{finding['issue']}*
                            
                            **Explanation:**  
                            {finding['explanation']}
                            
                            **Recommendation:**  
                            💡 *{finding['recommendation']}*
                            
                            ---
                            """, unsafe_allow_html=True)
                else:
                    st.error("Failed to load bug analysis.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ----------------- Tab 6: Security -----------------
    with tab_sec:
        st.subheader("Security Vulnerabilities & Secrets Analysis")
        with st.spinner("Running security assessment..."):
            try:
                sec_resp = requests.get(f"{API_URL}/repository/{owner}/{name}/security")
                if sec_resp.status_code == 200:
                    sec_list = sec_resp.json()
                    if not sec_list:
                        st.success("No hardcoded secrets or unsafe function patterns detected.")
                    else:
                        for finding in sec_list:
                            st.markdown(f"""
                            ### 🔒 Category: `{finding['category'].replace('_', ' ').title()}`
                            **File:** `{finding['file_path']}` (Line {finding['line']}) | **Confidence:** `{finding['confidence']}`
                            
                            **Pattern Name:** `{finding['pattern_name']}`
                            
                            **Masked Value:**  
                            {finding['masked_value']}
                            
                            **Recommendation:**  
                            💡 *{finding['recommendation']}*
                            
                            ---
                            """, unsafe_allow_html=True)
                else:
                    st.error("Failed to load security findings.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ----------------- Tab 7: GitHub Issues -----------------
    with tab_issues:
        st.subheader("GitHub Issue Impact Analysis")
        issue_number = st.number_input("GitHub Issue Number", min_value=1, step=1, value=1)
        analyze_issue_btn = st.button("🔍 Analyze Issue", use_container_width=True)
        
        if analyze_issue_btn:
            with st.spinner("Analyzing issue impact..."):
                try:
                    issue_resp = requests.post(f"{API_URL}/repository/{owner}/{name}/issues/{issue_number}")
                    if issue_resp.status_code == 200:
                        issue_data = issue_resp.json()
                        st.markdown(f"### 🎫 Issue #{issue_data['issue_number']}: {issue_data['title']}")
                        st.markdown(f"**State:** `{issue_data['state'].upper()}` | **Confidence:** `{issue_data['confidence']}`")
                        
                        st.markdown("#### Likely Affected Files")
                        for f in issue_data["likely_affected_files"]:
                            st.markdown(f"- 📁 `{f['file_path']}` (lines {f['start_line']} - {f['end_line']})")
                            
                        st.markdown("#### Probable Root Cause")
                        st.info(issue_data["probable_root_cause"])
                        
                        st.markdown("#### Related Components")
                        st.write(", ".join(issue_data["related_components"]) or "None identified")
                        
                        st.markdown("#### Full Analysis Details")
                        st.write(issue_data["full_analysis"])
                    else:
                        err = issue_resp.json().get("detail", "Issue not found")
                        st.error(err)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ----------------- Tab 8: Pull Requests -----------------
    with tab_prs:
        st.subheader("Pull Request Review Analysis")
        pr_number = st.number_input("Pull Request Number", min_value=1, step=1, value=1)
        analyze_pr_btn = st.button("🔍 Analyze Pull Request", use_container_width=True)
        
        if analyze_pr_btn:
            with st.spinner("Reviewing Pull Request..."):
                try:
                    pr_resp = requests.post(f"{API_URL}/repository/{owner}/{name}/pulls/{pr_number}")
                    if pr_resp.status_code == 200:
                        pr_data = pr_resp.json()
                        st.markdown(f"### 🔌 PR #{pr_data['pr_number']}: {pr_data['title']}")
                        st.markdown(f"**State:** `{pr_data['state'].upper()}`")
                        
                        st.markdown("#### Changed Files")
                        st.write(", ".join(pr_data["changed_files"]))
                        
                        st.markdown("#### Findings")
                        for finding in pr_data["findings"]:
                            cat_map = {
                                "code_quality": "badge-info",
                                "potential_bug": "badge-error",
                                "missing_tests": "badge-warning"
                            }
                            cat_class = cat_map.get(finding['category'], "badge-info")
                            file_loc = f" in `{finding['file_path']}`" if finding['file_path'] else ""
                            st.markdown(f"""
                            - <span class="badge {cat_class}">{finding['category'].replace('_', ' ').upper()}</span> **Confidence:** `{finding['confidence']}` {file_loc}
                              * {finding['description']}
                            """, unsafe_allow_html=True)
                            
                        st.markdown("#### Overall Assessment")
                        st.success(pr_data["overall_assessment"])
                    else:
                        err = pr_resp.json().get("detail", "Pull Request not found")
                        st.error(err)
                except Exception as e:
                    st.error(f"Error: {e}")
