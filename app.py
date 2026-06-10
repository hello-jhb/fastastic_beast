"""
app.py — v2 frontend.

Flow:
  1. Landing screen: scenario picker (4 cards, 2 active + 2 "coming soon").
  2. After picking a scenario, scoped chat workspace:
     - "← Back to scenarios" header
     - File uploader (auto-clears previous batch + resets SSOT)
     - Chat thread with the scenario-bound agent
     - SSOT panel in an expander

The agent does all the heavy lifting (classify, ingest, run scenario, answer
follow-ups). This file is just orchestration + presentation.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import ssot
import tools
from agent_loop import AgentSession, SCENARIO_CONFIG


# Which scenario tool to run deterministically per scenario key.
_SCENARIO_RUNNER = {
    "deal_review": tools.run_deal_review,
    "perf_vs_plan": tools.run_perf_vs_plan,
}


# =============================================================================
# Page config & global CSS
# =============================================================================

st.set_page_config(
    page_title="Fantastic Beast & Where to Find Them",
    page_icon="🏢",
    layout="wide",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, .stApp { font-family: "Inter", system-ui, sans-serif; }
  .block-container { padding-top: 2rem; max-width: 1100px; }

  /* Hero */
  .hero-title { font-size: 32px; font-weight: 700; margin-bottom: 4px; }
  .hero-sub   { font-size: 15px; color: #6b7280; margin-bottom: 24px; }

  /* Scenario cards */
  .scenario-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px;
    background: #ffffff;
    height: 100%;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .scenario-card.active:hover {
    border-color: #2563eb;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
  }
  .scenario-card.disabled { opacity: 0.55; background: #fafafa; }
  .scenario-card .label {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 12px;
  }
  .scenario-card .label.live    { background: #dcfce7; color: #166534; }
  .scenario-card .label.soon    { background: #f3f4f6; color: #6b7280; }
  .scenario-card .title  { font-size: 18px; font-weight: 600; margin-bottom: 6px; }
  .scenario-card .desc   { font-size: 13px; color: #4b5563; line-height: 1.5; min-height: 56px; }

  /* Scoped workspace header */
  .ws-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 18px; }
  .ws-title  { font-size: 22px; font-weight: 600; }
  .ws-scen   { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; }

  /* SSOT pills */
  .ssot-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    background: #eef2ff;
    color: #3730a3;
    font-size: 11px;
    font-weight: 500;
    margin: 2px 4px 2px 0;
  }

  /* Tool-trace items */
  .tool-trace {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 11px;
    color: #4b5563;
    margin: 2px 0;
  }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Session state
# =============================================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Default values for every session_state key the app uses. Defined as a single
# dict so we can both initialize on first load AND defensively fall back to
# defaults if any key is missing later (e.g. after a Streamlit error / rerun
# that somehow lost state).
_SESSION_DEFAULTS: dict = {
    "active_scenario":    None,
    "agent_session":      None,
    "uploaded_filenames": set(),
    "last_auto_message":  None,
    # Files whose auto-classification failed and are awaiting a user choice.
    # Shape: {filename: error_message}
    "pending_overrides":  {},
    # Set of batches (frozensets of filenames) we've already run the scenario for.
    "completed_batches":  set(),
}


def _ensure_session_state() -> None:
    """
    Initialize every session_state key that doesn't already exist.
    Called both at module load AND defensively at the top of every render
    function so a missing-attr error can never trigger from these keys.
    """
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            # Use a fresh copy of mutable defaults so all sessions don't share
            # the same dict/set instance.
            if isinstance(default, (set, dict, list)):
                st.session_state[key] = type(default)()
            else:
                st.session_state[key] = default


# Run once on module load — every script rerun re-executes this module
_ensure_session_state()


# Layer options the user can pick from the manual-override dropdown.
# Must match ssot.KNOWN_LAYERS exactly. Ordered: most common choices first.
_MANUAL_LAYER_OPTIONS = [
    "underwriting",
    "business_plan",
    "actuals_recent",
    "actuals_2020",
    "actuals_2021",
    "actuals_2022",
    "actuals_2023",
    "actuals_2024",
    "actuals_2025",
    "rent_roll",
    "debt",
]


# =============================================================================
# Helpers
# =============================================================================

def _wipe_uploads_and_reset_ssot() -> None:
    """Clean slate for a new analysis."""
    for p in UPLOAD_DIR.iterdir():
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    ssot.reset_ssot()
    st.session_state.uploaded_filenames = set()
    st.session_state.pending_overrides = {}
    st.session_state.completed_batches = set()


def _activate_scenario(scenario_key: str) -> None:
    """User clicked a scenario card — start a fresh session for it."""
    _wipe_uploads_and_reset_ssot()
    st.session_state.active_scenario = scenario_key
    st.session_state.agent_session = AgentSession(scenario_key)
    st.session_state.last_auto_message = None


def _back_to_landing() -> None:
    st.session_state.active_scenario = None
    st.session_state.agent_session = None


def _ssot_panel() -> None:
    """Show what's currently in SSOT."""
    summary = ssot.ssot_summary()
    layers = summary["layers_present"]
    files = summary["ingested_files"]

    if not layers and not files:
        st.caption("No files ingested yet.")
        return

    # Show last ingested timestamp — makes stale SSOT data immediately visible
    last_update = summary.get("updated_at")
    if last_update:
        from datetime import datetime, timezone
        try:
            ts = datetime.fromisoformat(last_update)
            age = datetime.now(timezone.utc) - ts
            hours = int(age.total_seconds() // 3600)
            age_str = f"{hours}h ago" if hours < 48 else f"{age.days}d ago"
            st.caption(f"Last ingested: {age_str}")
        except Exception:
            st.caption(f"Last ingested: {last_update[:10]}")

    st.markdown("**Layers in SSOT:**")
    if layers:
        st.markdown(
            " ".join(f'<span class="ssot-pill">{layer}</span>' for layer in layers),
            unsafe_allow_html=True,
        )
    else:
        st.caption("(none)")

    st.markdown("**Files ingested:**")
    if files:
        for f in files:
            st.markdown(f"- {f}")
    else:
        st.caption("(none)")

    # Catalog improvement suggestions from Pass 2 GPT gap-fill.
    # These are labels GPT found in the file that aren't in the catalog yet.
    # Adding them as aliases means the next file won't need GPT to find them.
    all_suggestions = []
    s = ssot.load_ssot()
    for layer_data in s.get("layers", {}).values():
        all_suggestions.extend(layer_data.get("catalog_suggestions", []))

    if all_suggestions:
        with st.expander(f"💡 {len(all_suggestions)} catalog alias suggestion(s)", expanded=False):
            st.caption(
                "GPT found these metrics under labels not in the catalog. "
                "Add them to Snapshot Metric.xlsx to avoid needing GPT for future files."
            )
            for s_ in all_suggestions:
                st.markdown(
                    f"**{s_['metric_name']}** — add alias: `{s_['found_as_label']}` "
                    f"(sheet: {s_.get('sheet', '?')})"
                )

    # Analyst Bundle — the reviewable run package: what Collie looked at,
    # what it believes, and what it refused to trust. The trust bridge.
    try:
        from analyst_bundle import build_analyst_bundle
        bundle = build_analyst_bundle("underwriting")
        if "error" not in bundle:
            ss = bundle.get("status_summary", {})
            n_issues = len(bundle.get("issues", []))
            n_verified = len(bundle.get("verified_facts", []))
            with st.expander(
                f"🔎 Analyst Bundle — {n_verified} verified / {n_issues} to review",
                expanded=False,
            ):
                metadata = bundle.get("run_metadata", {})
                if metadata:
                    st.markdown("**Run Metadata**")
                    st.json(metadata)

                wm = bundle.get("workbook_map", {})
                st.markdown("**Workbook Map** — which tabs Collie read vs skipped")
                st.caption(
                    f"{len(wm.get('all_sheets', []))} sheets · "
                    f"{len(wm.get('skipped_sheets', []))} skipped "
                    f"({', '.join(wm.get('skipped_sheets', [])[:6])}{'…' if len(wm.get('skipped_sheets', []))>6 else ''})"
                )
                content_roles = wm.get("content_roles") or {}
                authoritative_tabs = wm.get("authoritative_tabs") or {}
                if content_roles:
                    st.markdown("**Content Roles** — workbook mapper classifications")
                    st.dataframe(
                        [
                            {
                                "sheet": sheet,
                                "role": info.get("role"),
                                "confidence": info.get("confidence"),
                                "tier": info.get("implied_tier"),
                            }
                            for sheet, info in content_roles.items()
                        ],
                        use_container_width=True,
                    )
                if authoritative_tabs:
                    st.markdown("**Authoritative Tabs** — section-reader source map")
                    st.json(authoritative_tabs)

                st.markdown("**Status Summary** — QC health check")
                st.json(ss)

                identity_checks = bundle.get("identity_checks") or []
                if identity_checks:
                    st.markdown("**Identity Checks** — final reconciled values only")
                    st.dataframe(identity_checks, use_container_width=True)

                extraction_plan = bundle.get("extraction_plan") or []
                if extraction_plan:
                    with st.expander("Extraction Plan", expanded=False):
                        st.dataframe(
                            [
                                {
                                    "metric": row.get("metric"),
                                    "section": row.get("section"),
                                    "expected_roles": ", ".join(row.get("expected_source_roles") or []),
                                    "actual_sheet": (row.get("actual_source_used") or {}).get("sheet"),
                                    "actual_cell": (row.get("actual_source_used") or {}).get("cell"),
                                    "status": (row.get("actual_source_used") or {}).get("status"),
                                    "method": (row.get("actual_source_used") or {}).get("method"),
                                }
                                for row in extraction_plan
                            ],
                            use_container_width=True,
                        )

                bpr = bundle.get("business_plan_read", {})
                if any(bpr.values()):
                    st.markdown("**Business-Plan Read** — GPT interpretation (not facts)")
                    st.json({k: v for k, v in bpr.items() if v})

                st.markdown("**Issues / QC Flags** — what Collie refused to trust")
                issues = bundle.get("issues", [])
                if issues:
                    st.dataframe(
                        [{"metric": r["metric"], "status": r["status"],
                          "value": r["value"], "source": f"{r.get('source_sheet')}!{r.get('source_cell')}"}
                         for r in issues],
                        use_container_width=True,
                    )
                else:
                    st.caption("None — every bounded metric verified.")

                st.markdown("**Verified Facts** — what Collie believes, with provenance")
                st.dataframe(
                    [{"metric": r["metric"], "value": r["value"],
                      "period": r.get("period"), "status": r["status"],
                      "source": f"{r.get('source_sheet')}!{r.get('source_cell')}"}
                     for r in bundle.get("verified_facts", [])],
                    use_container_width=True,
                )
                source_audit = bundle.get("source_audit") or []
                if source_audit:
                    with st.expander("Source Audit", expanded=False):
                        st.json(source_audit)
                if bundle.get("bundle_path"):
                    st.caption(f"Saved: {bundle['bundle_path']}")
    except Exception:
        pass


# =============================================================================
# Landing view — scenario picker
# =============================================================================

def render_landing() -> None:
    _ensure_session_state()  # defensive safety net
    st.markdown(
        '<div class="hero-title">Fantastic Beast & Where to Find Them</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Pick an analysis to start. Each scenario is a '
        'scoped workspace — upload the relevant files and chat with the agent.</div>',
        unsafe_allow_html=True,
    )

    # Card grid: 2 columns x 2 rows
    cards = [
        {
            "key": "deal_review",
            "label": "live",
            "title": "Deal Analysis",
            "desc": "Summarize an acquisition from a single underwriting model. "
                    "Going-in basis, NOI, IRR, exit value, debt terms.",
            "active": True,
        },
        {
            "key": "perf_vs_plan",
            "label": "live",
            "title": "Performance Analysis",
            "desc": "Compare actuals against the underwriting or business plan. "
                    "Year-by-year variance with driver attribution.",
            "active": True,
        },
        {
            "key": "lease_review",
            "label": "soon",
            "title": "Lease Review",
            "desc": "Reconcile tenant-level data between leases and rent rolls. "
                    "Flag discrepancies in term, base rent, escalations.",
            "active": False,
        },
        {
            "key": "debt_analysis",
            "label": "soon",
            "title": "Debt Analysis",
            "desc": "DSCR, debt yield, LTV against loan covenants. "
                    "Refinance and maturity outlook.",
            "active": False,
        },
    ]

    row1 = st.columns(2, gap="medium")
    row2 = st.columns(2, gap="medium")

    for card, col in zip(cards, [*row1, *row2]):
        with col:
            cls = "scenario-card active" if card["active"] else "scenario-card disabled"
            label_cls = "live" if card["active"] else "soon"
            label_text = "Available" if card["active"] else "Coming soon"

            st.markdown(
                f"""
<div class="{cls}">
  <span class="label {label_cls}">{label_text}</span>
  <div class="title">{card['title']}</div>
  <div class="desc">{card['desc']}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            if card["active"]:
                st.button(
                    f"Start →",
                    key=f"start_{card['key']}",
                    on_click=_activate_scenario,
                    args=(card["key"],),
                    use_container_width=True,
                )
            else:
                st.button(
                    "Not available yet",
                    key=f"disabled_{card['key']}",
                    disabled=True,
                    use_container_width=True,
                )


# =============================================================================
# Scenario view — file uploader + chat
# =============================================================================

def render_scenario() -> None:
    # Defensive: ensure all session_state keys exist before any access.
    # Streamlit *should* keep them across reruns, but a partial-execution
    # error can occasionally leave state in a half-initialized place.
    _ensure_session_state()

    scenario_key = st.session_state.active_scenario
    cfg = SCENARIO_CONFIG[scenario_key]
    agent: AgentSession = st.session_state.agent_session

    # Header
    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            f'<div class="ws-header"><div>'
            f'<div class="ws-scen">{cfg["display_name"]}</div>'
            f'<div class="ws-title">Workspace</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.button("← Back to scenarios", on_click=_back_to_landing, use_container_width=True)

    st.divider()

    # File uploader
    uploaded = st.file_uploader(
        "Upload files for this analysis",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        key=f"upload_{scenario_key}",
    )

    # Detect a new upload batch (different filenames than what we've already
    # processed in this session). Save the new files to disk.
    new_files = []
    if uploaded:
        current_names = {f.name for f in uploaded}
        if current_names != st.session_state.uploaded_filenames:
            new_files = [f for f in uploaded if f.name not in st.session_state.uploaded_filenames]
            for uf in new_files:
                (UPLOAD_DIR / uf.name).write_bytes(uf.getbuffer())
            st.session_state.uploaded_filenames = current_names

    # SSOT panel
    with st.expander("📂 SSOT — Asset record", expanded=False):
        _ssot_panel()

    st.divider()

    # Replay chat history so the workspace looks consistent across reruns.
    for m in agent.display_messages():
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ---------------------------------------------------------------------
    # Deterministic orchestration — 3 phases, each idempotent across reruns.
    # ---------------------------------------------------------------------

    # Phase 1: ingest any new files (queues manual-override candidates).
    if new_files:
        _ingest_new_files(new_files)

    # Phase 2: if files need manual classification, show the override form
    # and stop — we can't run the scenario until layers are resolved.
    if st.session_state.get("pending_overrides"):
        _render_manual_override_ui()
        # Still allow follow-up Q&A while waiting on overrides
        user_input = st.chat_input("Ask a follow-up question...")
        _handle_chat_input(agent, user_input)
        return

    # Phase 3: run the scenario if we haven't already for this batch.
    if st.session_state.uploaded_filenames:
        batch_id = frozenset(st.session_state.uploaded_filenames)
        if batch_id not in st.session_state.completed_batches:
            _run_scenario_for_batch(agent, scenario_key)

    # Deep-dive buttons — only show for deal_review scenario after the memo is generated
    if (
        scenario_key == "deal_review"
        and st.session_state.uploaded_filenames
        and frozenset(st.session_state.uploaded_filenames) in st.session_state.completed_batches
    ):
        _render_deep_dive_buttons(agent)

    # User chat input — this is where the agent earns its keep (Q&A).
    user_input = st.chat_input("Ask a follow-up question...")
    _handle_chat_input(agent, user_input)


def _render_deep_dive_buttons(agent: AgentSession) -> None:
    """Render the 5 deep-dive buttons above the chat input."""
    st.markdown(
        '<div style="margin-top:8px; font-size:11px; color:#6b7280; '
        'text-transform:uppercase; letter-spacing:0.06em;">Drill into a section</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    button_specs = [
        ("capital_structure", "Capital Structure"),
        ("cash_flow",         "Cash Flow / NOI"),
        ("return_profile",    "Return Profile"),
        ("capex_plan",        "CapEx Plan"),
        ("key_risks",         "Key Risks"),
    ]
    for i, (key, label) in enumerate(button_specs):
        with cols[i]:
            if st.button(label, key=f"dd_{key}", use_container_width=True):
                _run_deep_dive(agent, key, label)


def _run_deep_dive(agent: AgentSession, key: str, label: str) -> None:
    """Execute a deep-dive scenario and seed the result into the chat history."""
    from scenarios.deep_dives import run_deep_dive

    pseudo_user_msg = f"📊 {label} deep dive"
    with st.chat_message("user"):
        st.markdown(pseudo_user_msg)
    with st.chat_message("assistant"):
        with st.spinner(f"Generating {label}..."):
            result = run_deep_dive(key)
        if "error" in result:
            st.error(result["error"])
            return
        st.markdown(result["narrative"])

    # Seed into agent message history so follow-up Q&A can reference it
    agent.messages.append({"role": "user",      "content": pseudo_user_msg})
    agent.messages.append({"role": "assistant", "content": result["narrative"]})


def _handle_chat_input(agent: AgentSession, user_input: str | None) -> None:
    if not user_input:
        return
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = agent.send(user_input)
        st.markdown(reply)
    _render_tool_trace(agent)


_SCENARIO_DEFAULT_LAYER: dict[str, str] = {
    # When a file can't be classified by name, fall back to the layer that
    # makes most sense for the active scenario.
    "deal_review":  "underwriting",
    "perf_vs_plan": "actuals_recent",
}


def _ingest_new_files(new_files: list) -> None:
    """
    Phase 1: ingest each newly-uploaded file.

    Classification strategy (in order):
      1. Auto-classify from filename (proforma, financial statement, etc.)
      2. Scenario-aware fallback — if filename gives no signal, use the layer
         that matches the active scenario (deal_review → underwriting, etc.)
      3. Only ask the user to manually classify if the scenario itself is unclear.

    Files that still can't be resolved are stashed in pending_overrides.
    """
    pseudo_user_msg = "Uploaded: " + ", ".join(sorted(f.name for f in new_files))
    with st.chat_message("user"):
        st.markdown(pseudo_user_msg)

    failed_to_classify: dict[str, str] = {}
    scenario_key = st.session_state.active_scenario
    scenario_fallback = _SCENARIO_DEFAULT_LAYER.get(scenario_key)

    with st.chat_message("assistant"):
        with st.status("Ingesting files...", expanded=True) as status:
            for uf in new_files:
                status.update(label=f"Ingesting {uf.name}...")
                result = tools.ingest_to_ssot(uf.name)

                if result.get("needs_manual_classification") and scenario_fallback:
                    # Filename gave no signal — use the scenario context as the layer.
                    result = tools.ingest_to_ssot_with_layer(uf.name, scenario_fallback)
                    if "error" not in result:
                        st.markdown(
                            f"✅ **{uf.name}** → `{scenario_fallback}` "
                            f"(auto-assigned from scenario — "
                            f"{result['metric_count']} metrics extracted)"
                        )
                    else:
                        failed_to_classify[uf.name] = result.get("error", "")
                        st.markdown(f"❌ **{uf.name}** — {result['error']}")

                elif result.get("needs_manual_classification"):
                    failed_to_classify[uf.name] = result.get("error", "")
                    st.markdown(f"⚠️ **{uf.name}** — needs manual classification")

                elif "error" in result:
                    st.markdown(f"❌ **{uf.name}** — {result['error']}")

                else:
                    insight_status = result.get("insight_pass", "?")
                    insight_emoji = "🤖" if "completed" in insight_status else "⚠️"
                    st.markdown(
                        f"✅ **{uf.name}** → `{result['layer']}` "
                        f"({result['metric_count']} metrics extracted) "
                        f"· {insight_emoji} Pass 2: {insight_status}"
                    )

            if failed_to_classify:
                status.update(
                    label=f"{len(failed_to_classify)} file(s) need manual classification",
                    state="error",
                )
            else:
                status.update(label="Ingest complete", state="complete")

    if failed_to_classify:
        st.session_state.pending_overrides = failed_to_classify


def _run_scenario_for_batch(agent: AgentSession, scenario_key: str) -> None:
    """
    Phase 3: readiness check + scenario run. Idempotent: marks the batch as
    completed when done so reruns don't repeat the work.
    """
    pseudo_user_msg = "Uploaded: " + ", ".join(sorted(st.session_state.uploaded_filenames))

    with st.chat_message("assistant"):
        with st.status("Generating analysis...", expanded=True) as status:
            readiness = tools.check_scenario_ready(scenario_key)
            if not readiness.get("ready"):
                status.update(label="More data needed", state="error")
                missing_msg = (
                    f"**Can't run {SCENARIO_CONFIG[scenario_key]['display_name']} yet.**\n\n"
                    f"{readiness.get('reason', 'Missing required layers.')}\n\n"
                    f"- Layers in SSOT now: `{readiness.get('layers_present', [])}`\n"
                    f"- Example of what's still needed: `{readiness.get('example_missing', [])}`"
                )
                st.markdown(missing_msg)
                _seed_agent_history(agent, pseudo_user_msg, missing_msg, [], None)
                st.session_state.completed_batches.add(frozenset(st.session_state.uploaded_filenames))
                return

            runner = _SCENARIO_RUNNER[scenario_key]
            scenario_result = runner()

            if "error" in scenario_result:
                status.update(label="Analysis failed", state="error")
                err_msg = f"**Couldn't generate the analysis:** {scenario_result['error']}"
                st.markdown(err_msg)
                _seed_agent_history(agent, pseudo_user_msg, err_msg, [], None)
                st.session_state.completed_batches.add(frozenset(st.session_state.uploaded_filenames))
                return

            status.update(label="Done", state="complete")

        st.markdown(scenario_result["narrative"])

    st.session_state.completed_batches.add(frozenset(st.session_state.uploaded_filenames))
    _seed_agent_history(agent, pseudo_user_msg, scenario_result["narrative"], [], scenario_result)


def _render_manual_override_ui() -> None:
    """Show a form letting the user classify any files that auto-classification missed."""
    scenario_key = st.session_state.active_scenario
    st.divider()
    st.markdown("### Manual classification")
    st.caption(
        "These files couldn't be classified by name. Tell me what each one is, "
        "and I'll ingest them into the right SSOT layer."
    )

    # Suggest a sensible default based on the active scenario.
    default_layer = {
        "deal_review": "underwriting",
        "perf_vs_plan": "actuals_recent",
    }.get(scenario_key, "underwriting")

    with st.form(key="manual_override_form"):
        choices: dict[str, str] = {}
        for filename in sorted(st.session_state.get("pending_overrides", {})):
            choices[filename] = st.selectbox(
                f"📄 {filename}",
                options=_MANUAL_LAYER_OPTIONS,
                index=_MANUAL_LAYER_OPTIONS.index(default_layer),
                key=f"override_{filename}",
            )
        submitted = st.form_submit_button("Ingest with these layers", type="primary")

    if submitted:
        # Run the override ingests.
        with st.status("Ingesting with manual layers...", expanded=True) as status:
            for filename, layer in choices.items():
                status.update(label=f"Ingesting {filename} as {layer}...")
                result = tools.ingest_to_ssot_with_layer(filename, layer)
                if "error" in result:
                    st.error(f"❌ {filename}: {result['error']}")
                else:
                    st.markdown(f"✅ **{filename}** → `{layer}` ({result['metric_count']} metrics)")
            status.update(label="Done", state="complete")

        # Clear the override queue and invalidate the completed-batches cache so
        # the scenario runs on the next rerun (which happens automatically after form submit).
        st.session_state.pending_overrides = {}
        st.session_state.completed_batches = set()
        st.rerun()


def _seed_agent_history(
    agent: AgentSession,
    user_msg: str,
    assistant_msg: str,
    ingest_results: list[dict],
    scenario_result: dict | None,
) -> None:
    """
    Append the work-just-done into the agent's message history so it has full
    context for any follow-up Q&A. The agent won't re-run ingest or scenario
    because it can see they already happened.
    """
    # Build a tool-summary line so the agent knows what's in SSOT.
    layers_now = ssot.list_layers()
    context_note = (
        f"[System: I (the host app) already ingested the uploaded files and "
        f"ran the scenario. Current SSOT layers: {layers_now}. "
        f"Do not call ingest_to_ssot or run_<scenario> again for these files. "
        f"For follow-up questions, use get_layer_details or get_ssot_summary.]"
    )
    agent.messages.append({"role": "user", "content": user_msg})
    agent.messages.append({"role": "assistant", "content": assistant_msg})
    agent.messages.append({"role": "user", "content": context_note})
    # Have the model acknowledge so the next true user message lands cleanly.
    agent.messages.append({"role": "assistant", "content": "Acknowledged. Ready for follow-up questions."})


def _render_tool_trace(agent: AgentSession) -> None:
    """Show what tools the agent called in its last turn (for transparency)."""
    if not agent.last_tool_calls:
        return
    with st.expander(f"🔧 Tool calls this turn ({len(agent.last_tool_calls)})", expanded=False):
        for tc in agent.last_tool_calls:
            args_preview = ", ".join(f"{k}={v!r}" for k, v in tc["arguments"].items())
            st.markdown(
                f'<div class="tool-trace">→ <b>{tc["name"]}</b>({args_preview})</div>',
                unsafe_allow_html=True,
            )
            result = tc["result"]
            if isinstance(result, dict) and "error" in result:
                st.markdown(f'<div class="tool-trace">  ❌ {result["error"]}</div>', unsafe_allow_html=True)
            elif isinstance(result, dict):
                # Compact summary based on result keys
                preview_keys = [k for k in ("filename", "layer", "metric_count", "layers_now_present",
                                            "files", "ready", "narrative") if k in result]
                if "narrative" in preview_keys:
                    st.markdown('<div class="tool-trace">  ✓ narrative generated</div>', unsafe_allow_html=True)
                else:
                    preview = {k: result[k] for k in preview_keys}
                    st.markdown(f'<div class="tool-trace">  ✓ {preview}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tool-trace">  ✓ {result}</div>', unsafe_allow_html=True)


# =============================================================================
# Router
# =============================================================================

if st.session_state.active_scenario is None:
    render_landing()
else:
    render_scenario()
