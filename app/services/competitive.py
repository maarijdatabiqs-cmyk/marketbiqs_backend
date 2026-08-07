import asyncio
import json
import logging
import re
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Agency,
    ClientBrand,
    Competitor,
    FeatureComparison,
    FeatureTicket,
    GapReport,
    GoalAlert,
    Integration,
    JobStatus,
    ProductFeature,
    TrackingJob,
)
from app.services import ai as ai_service
from app.services import jira as jira_service
from app.services.reports import generate_client_report
from app.services.tracking import scrape_website, serp_visibility

logger = logging.getLogger("marketbiqs.competitive")


def _clip(value: str | None, max_len: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _level_label(value: str | None, default: str = "medium", *, max_len: int = 40) -> str:
    """Normalize AI severity labels; allow short free text up to max_len."""
    raw = _as_str(value, default).strip()
    lowered = raw.lower()
    if lowered in {"low", "medium", "high", "critical"}:
        return lowered
    if not raw:
        return default
    return _clip(raw, max_len)


_WEAK_COMPARISON_MARKERS = (
    "none",
    "n/a",
    "does not have a similar feature",
    "do not have a similar feature",
    "both companies have similar features",
    "continue to enhance and promote",
    "continue to monitor and improve",
    "no similar feature",
    "similar features",
)


def _as_str(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def _as_int(value, default: int | None = None) -> int | None:
    """AI returns story points as 5, '5', '5 points', or '3-5' — keep the first integer."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"\d+", _as_str(value))
    return int(match.group()) if match else default


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def _is_generic_text(value) -> bool:
    text = _as_str(value).strip().lower()
    if len(text) < 12:
        return True
    return any(marker in text for marker in _WEAK_COMPARISON_MARKERS)


def _normalize_comparison_row(row: dict, client_name: str, competitor_name: str) -> dict | None:
    feature_name = _as_str(row.get("feature_name")).strip()
    if not feature_name:
        return None

    our_status = _as_str(row.get("our_status"), "parity").strip().lower()
    competitor_status = _as_str(row.get("competitor_status"), "parity").strip().lower()
    status_map = {
        "lead": "leading",
        "leading": "leading",
        "strong": "leading",
        "has": "leading",
        "available": "leading",
        "parity": "parity",
        "equal": "parity",
        "similar": "parity",
        "lagging": "lagging",
        "weak": "lagging",
        "missing": "lagging",
        "none": "lagging",
        "absent": "lagging",
        "behind": "lagging",
    }
    our_status = status_map.get(our_status, "parity")
    competitor_status = status_map.get(competitor_status, "parity")

    note = _as_str(row.get("note")).strip()
    how_leads = _as_str(row.get("how_competitor_leads")).strip()
    how_improve = _as_str(row.get("how_to_improve")).strip()

    if _is_generic_text(note):
        note = f"{competitor_name} vs {client_name} on {feature_name}: rival posture is {competitor_status}, yours is {our_status}."
    if _is_generic_text(how_leads):
        how_leads = f"{competitor_name} is positioned as {competitor_status} on {feature_name} in public materials and product packaging."
    if _is_generic_text(how_improve):
        how_improve = f"Ship a clearer {feature_name} offer, proof points, and sales narrative to close the gap with {competitor_name}."

    citations = row.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    clean_citations = []
    for c in citations[:4]:
        if isinstance(c, dict) and (c.get("url") or c.get("snippet")):
            clean_citations.append(
                {
                    "url": _as_str(c.get("url")),
                    "snippet": _as_str(c.get("snippet"))[:400],
                    "source": _as_str(c.get("source"), "web"),
                }
            )

    try:
        confidence = float(row.get("confidence_score") or 0.55)
    except (TypeError, ValueError):
        confidence = 0.55
    confidence = max(0.0, min(1.0, confidence))
    if clean_citations:
        confidence = max(confidence, 0.65)
    evidence = _as_str(row.get("evidence_strength"), "medium").strip().lower()
    if evidence not in {"low", "medium", "high"}:
        evidence = "medium" if clean_citations else "low"

    contested = competitor_status == "leading" or our_status == "lagging"

    return {
        "feature_name": feature_name,
        "category": _as_str(row.get("category"), "General").strip() or "General",
        "our_status": our_status,
        "competitor_status": competitor_status,
        "note": note,
        "how_competitor_leads": how_leads,
        "how_to_improve": how_improve,
        "citations": clean_citations,
        "confidence_score": confidence,
        "evidence_strength": evidence,
        "is_contested_move": contested,
    }


async def _generate_competitor_comparisons(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    features: list[ProductFeature],
    competitor: Competitor,
) -> dict:
    result = await ai_service.structured_json(
        db,
        agency.id,
        (
            "You are a senior competitive strategist writing UpdatePromise/Databiqs-quality comparison rows. "
            "Return JSON: {competitor_name, rows:[{feature_name, category, our_status, competitor_status, note, how_competitor_leads, how_to_improve, confidence_score, evidence_strength, citations:[{url, snippet, source}]}]}. "
            "Rules:\n"
            "1) Only include contested features where the rival is leading, we are lagging, or parity is commercially dangerous.\n"
            "2) DO NOT include rows where we clearly lead and the rival lags.\n"
            "3) our_status/competitor_status must be leading|parity|lagging.\n"
            "4) note, how_competitor_leads, and how_to_improve must each be specific (1-3 sentences), concrete, and actionable.\n"
            "5) NEVER write: None, N/A, 'does not have a similar feature', 'both companies have similar features', "
            "'continue to enhance and promote', or 'continue to monitor and improve'.\n"
            "6) how_competitor_leads must explain buyer perception, GTM, packaging, workflow fit, or brand equity.\n"
            "7) how_to_improve must give a concrete counter-move (product packaging, proof, pricing page, demo narrative, content).\n"
            "8) Include citations with url + short snippet whenever possible from competitor website/features.\n"
            "9) confidence_score 0-1 and evidence_strength low|medium|high.\n"
            "10) Produce 3-6 high-signal rows only."
        ),
        json.dumps(
            {
                "client": {
                    "name": client.name,
                    "industry": client.industry,
                    "niche": client.niche,
                    "tagline": client.tagline,
                    "goals": client.goals or [],
                    "features": [
                        {"name": f.name, "category": f.category, "description": f.description} for f in features
                    ],
                },
                "competitor": {
                    "name": competitor.name,
                    "website": competitor.website,
                    "tagline": competitor.tagline,
                    "description": competitor.description,
                    "overlap_score": competitor.overlap_score,
                    "threat_level": competitor.threat_level,
                    "features": competitor.feature_list or [],
                },
            }
        )[:12000],
        temperature=0.4,
    )

    rows = result.get("rows") or []
    cleaned_rows = []
    for row in rows:
        cleaned = _normalize_comparison_row(row, client.name, competitor.name)
        if cleaned:
            cleaned_rows.append(cleaned)

    if len(cleaned_rows) < 2:
        repair = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Rewrite competitive comparison rows. Prioritize where the rival threatens the client. "
                "Return JSON {rows:[...] } with the same schema and strict anti-filler rules. "
                "Every how_competitor_leads and how_to_improve must be specific strategy language."
            ),
            json.dumps(
                {
                    "competitor_name": competitor.name,
                    "client_name": client.name,
                    "client_features": [f.name for f in features],
                    "competitor_features": [
                        (f.get("name") if isinstance(f, dict) else str(f)) for f in (competitor.feature_list or [])
                    ],
                    "weak_draft_rows": rows,
                }
            )[:10000],
            temperature=0.45,
        )
        for row in repair.get("rows") or []:
            cleaned = _normalize_comparison_row(row, client.name, competitor.name)
            if cleaned:
                cleaned_rows.append(cleaned)

    if not cleaned_rows:
        rival_feats = []
        for f in competitor.feature_list or []:
            if isinstance(f, dict) and f.get("name"):
                rival_feats.append(_as_str(f.get("name")))
            elif isinstance(f, str) and f.strip():
                rival_feats.append(f.strip())
        client_names = {f.name.lower() for f in features}
        for name in rival_feats[:5]:
            cleaned_rows.append(
                _normalize_comparison_row(
                    {
                        "feature_name": name,
                        "category": "Competitive",
                        "our_status": "lagging" if name.lower() not in client_names else "parity",
                        "competitor_status": "leading",
                        "note": f"{competitor.name} publicly emphasizes {name}.",
                        "how_competitor_leads": f"{competitor.name} markets {name} as a core differentiator.",
                        "how_to_improve": f"Package and prove a {name} response that sales can demo against {competitor.name}.",
                        "confidence_score": 0.55,
                        "evidence_strength": "medium",
                        "citations": [
                            {
                                "url": competitor.website or "",
                                "snippet": (competitor.evidence_snippet or competitor.description or name)[:280],
                                "source": "website",
                            }
                        ]
                        if competitor.website
                        else [],
                    },
                    client.name,
                    competitor.name,
                )
            )
        for feat in features[:4]:
            if any(r and r.get("feature_name", "").lower() == feat.name.lower() for r in cleaned_rows if r):
                continue
            cleaned_rows.append(
                _normalize_comparison_row(
                    {
                        "feature_name": feat.name,
                        "category": feat.category or "General",
                        "our_status": "parity",
                        "competitor_status": "leading",
                        "note": f"Compare {feat.name} depth vs {competitor.name}.",
                        "how_competitor_leads": f"{competitor.name} may out-package {feat.name} in buyer conversations.",
                        "how_to_improve": f"Tighten messaging, proof, and demo narrative for {feat.name}.",
                        "confidence_score": 0.5,
                        "evidence_strength": "low",
                    },
                    client.name,
                    competitor.name,
                )
            )
        cleaned_rows = [r for r in cleaned_rows if r]

    deduped: list[dict] = []
    seen: set[str] = set()
    for row in cleaned_rows:
        key = _as_str(row["feature_name"]).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return {"competitor_name": competitor.name, "rows": deduped[:8]}


def _extract_features_from_markdown(markdown: str, limit: int = 12) -> list[dict]:
    features: list[dict] = []
    seen: set[str] = set()
    skip = {"contact us", "home", "about", "privacy", "terms", "blog", "careers", "login", "sign in"}
    for raw in (markdown or "").splitlines():
        line = raw.strip().lstrip("#*-• ").strip()
        if not line or len(line) < 3 or len(line) > 80:
            continue
        if "http" in line.lower() or line.startswith("["):
            continue
        words = line.split()
        if len(words) > 8:
            continue
        lowered = line.lower()
        if lowered in seen or lowered in skip:
            continue
        seen.add(lowered)
        features.append(
            {
                "name": line,
                "category": "Capability",
                "description": (
                    f"{line} is something this company already offers. "
                    f"In simple terms, it is a capability they promote publicly on their website. "
                    f"Customers can ask for this as part of what the brand sells or delivers today."
                ),
            }
        )
        if len(features) >= limit:
            break
    return features



_FEATURE_DESC_PROMPT = (
    "For each feature, write a plain-English description a non-technical agency user can understand. "
    "Rules for every description:\n"
    "1) Exactly 2–3 short sentences.\n"
    "2) Explain what the customer gets / what problem it solves — not buzzwords.\n"
    "3) Avoid jargon like production-grade, demoware, architecture-first, hyperscale, MLOps, "
    "unless you immediately explain it in everyday words.\n"
    "4) Do not repeat only the feature name. Do not write marketing slogans.\n"
    "5) Keep names as given; only rewrite descriptions.\n"
    "Return JSON: {features:[{name, category, description}]}."
)


def _feature_description_is_thin(name: str, description: str) -> bool:
    name = _as_str(name).strip()
    desc = _as_str(description).strip()
    if not desc:
        return True
    if desc.lower() == name.lower():
        return True
    if len(desc) < 90:
        return True
    # slogan-ish one-liners with little explanation
    if desc.count(".") == 0 and len(desc) < 140:
        return True
    return False


def _fallback_plain_feature_description(name: str, category: str, description: str, client_name: str) -> str:
    name = _as_str(name).strip() or "This capability"
    category = _as_str(category).strip() or "General"
    raw = _as_str(description).strip()
    soft = raw or name
    replacements = (
        ("production-grade ai, not demoware", "AI that is ready for real day-to-day business use — not just a flashy demo"),
        ("production grade ai, not demoware", "AI that is ready for real day-to-day business use — not just a flashy demo"),
        ("production-grade", "ready for real day-to-day business use"),
        ("demoware", "a demo that looks good but is not ready for real work"),
        ("architecture-first thinking", "planning the system carefully before building anything"),
        ("architecture-first", "planned carefully before building"),
        ("enterprise-grade", "built for larger companies"),
        ("end-to-end", "handled from start to finish"),
        ("cutting-edge", "up-to-date"),
        ("state-of-the-art", "modern"),
        ("ai-powered", "using AI to help"),
        ("ml-powered", "using machine learning to help"),
    )
    lowered = soft
    for a, b in replacements:
        idx = lowered.lower().find(a.lower())
        while idx >= 0:
            lowered = lowered[:idx] + b + lowered[idx + len(a) :]
            idx = lowered.lower().find(a.lower(), idx + len(b))
    soft = " ".join(lowered.split())
    if soft.lower() == name.lower() or len(soft) < 40:
        soft = f"customers can use {name} as part of what {client_name or 'the brand'} delivers today"
    cat_bit = f" ({category})" if category and category.lower() not in {"general", "capability"} else ""
    mid = soft[0].lower() + soft[1:] if soft else "customers can use this today"
    return (
        f"{name} is something {client_name or 'this brand'} already offers{cat_bit}. "
        f"In simple terms, {mid}{'' if mid.endswith('.') else '.'} "
        f"This is part of their current offering — not a future idea."
    )


async def clarify_feature_descriptions(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    features: list[ProductFeature] | None = None,
) -> list[ProductFeature]:
    """Rewrite thin/jargon feature descriptions into plain 2–3 sentence English."""
    if features is None:
        features = (
            await db.execute(
                select(ProductFeature).where(
                    ProductFeature.client_id == client.id,
                    ProductFeature.agency_id == agency.id,
                    ProductFeature.is_wishlisted.is_(False),
                )
            )
        ).scalars().all()

    owned = [f for f in features if not f.is_wishlisted]
    if not owned:
        return owned

    thin = [f for f in owned if _feature_description_is_thin(f.name, f.description or "")]
    # Always clarify thin ones; if most are thin, rewrite the whole set for consistency
    targets = owned if len(thin) >= max(1, len(owned) // 2) else thin
    if not targets:
        return owned

    payload = [
        {"name": f.name, "category": f.category or "General", "description": f.description or ""}
        for f in targets
    ]
    rewritten = await ai_service.structured_json(
        db,
        agency.id,
        _FEATURE_DESC_PROMPT
        + f" Company name: {client.name}. Industry: {client.industry or 'unknown'}.",
        json.dumps({"features": payload})[:6000],
        temperature=0.2,
    )
    by_name: dict[str, dict] = {}
    for item in rewritten.get("features") if isinstance(rewritten.get("features"), list) else []:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("name")).strip().lower()
        if key:
            by_name[key] = item

    for feat in targets:
        item = by_name.get(_as_str(feat.name).lower())
        new_desc = _as_str(item.get("description")) if item else ""
        if item and item.get("category"):
            feat.category = _as_str(item.get("category") or feat.category, "General")
        if _feature_description_is_thin(feat.name, new_desc):
            new_desc = _fallback_plain_feature_description(
                feat.name, feat.category or "General", feat.description or new_desc, client.name
            )
        feat.description = new_desc

    await db.flush()
    return owned



# Hyperscalers, Big 4, mega SIs, and platform giants — not niche peer rivals.
_GLOBAL_RIVAL_BLOCKLIST = {
    "accenture", "ibm", "ibm watson", "watson", "microsoft", "microsoft ai", "microsoft azure",
    "azure", "google", "google cloud", "google cloud ai", "google ai", "dialogflow", "amazon",
    "aws", "amazon web services", "oracle", "oracle ai", "sap", "sap leonardo", "deloitte",
    "pwc", "ey", "ernst & young", "kpmg", "cognizant", "infosys", "capgemini", "tcs",
    "tata consultancy", "wipro", "meta", "openai", "anthropic", "salesforce", "adobe",
    "nvidia", "mckinsey", "bain", "bcg", "boston consulting", "slalom", "thoughtworks",
    "manychat", "converse.ai", "inbenta",
}

_GLOBAL_DOMAIN_BLOCKLIST = {
    "accenture.com", "ibm.com", "microsoft.com", "azure.microsoft.com", "google.com",
    "cloud.google.com", "dialogflow.cloud.google.com", "amazon.com", "aws.amazon.com",
    "oracle.com", "sap.com", "deloitte.com", "pwc.com", "ey.com", "kpmg.com",
    "cognizant.com", "infosys.com", "capgemini.com", "tcs.com", "wipro.com",
    "openai.com", "anthropic.com", "salesforce.com", "adobe.com", "nvidia.com",
    "mckinsey.com", "bain.com", "bcg.com", "slalom.com", "thoughtworks.com",
    "manychat.com", "converse.ai", "inbenta.com",
}


def _domain_of(url: str) -> str:
    raw = _as_str(url).strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        from urllib.parse import urlparse

        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_website(url: str | None) -> str | None:
    """Store absolute https URLs only; drop junk that cannot open in a browser."""
    raw = _as_str(url).strip()
    if not raw:
        return None
    # Reject placeholders / obvious non-URLs
    lowered = raw.lower()
    if lowered in {"n/a", "na", "none", "null", "-", "tbd", "unknown"}:
        return None
    if " " in raw or "\n" in raw:
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    elif not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    host = _domain_of(raw)
    if not host or "." not in host:
        return None
    # Reject bare TLDs / IP-less junk hosts
    if host.count(".") < 1 or host.endswith("."):
        return None
    return raw.rstrip("/")


# Country / market vocabulary for local-scope geo filtering
_COUNTRY_ALIASES: dict[str, set[str]] = {
    "pakistan": {
        "pakistan", "pakistani", "pk", "pak",
        "karachi", "lahore", "islamabad", "rawalpindi", "faisalabad",
        "multan", "peshawar", "sialkot", "gujranwala", "quetta",
    },
    "india": {
        "india", "indian", "bharat",
        "mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "hyderabad",
        "chennai", "pune", "noida", "gurgaon", "gurugram", "kolkata", "ahmedabad",
    },
    "singapore": {"singapore", "singaporean", "sg"},
    "uae": {
        "uae", "united arab emirates", "dubai", "abu dhabi", "abudhabi", "sharjah",
        "emirates",
    },
    "saudi arabia": {"saudi", "saudi arabia", "ksa", "riyadh", "jeddah", "dammam"},
    "united states": {
        "united states", "usa", "u.s.", "u.s.a", "america", "american",
        "california", "new york", "texas", "silicon valley",
    },
    "united kingdom": {"united kingdom", "uk", "u.k.", "britain", "british", "london", "england"},
    "canada": {"canada", "canadian", "toronto", "vancouver", "montreal", "ontario", "mississauga", "british columbia"},
    "australia": {"australia", "australian", "sydney", "melbourne"},
    "germany": {"germany", "german", "berlin", "munich"},
    "bangladesh": {"bangladesh", "bangladeshi", "dhaka"},
    "china": {"china", "chinese", "beijing", "shanghai", "shenzhen"},
}

_COUNTRY_TLDS: dict[str, set[str]] = {
    "pakistan": {".pk"},
    "india": {".in"},
    "singapore": {".sg"},
    "uae": {".ae"},
    "saudi arabia": {".sa"},
    "united kingdom": {".uk", ".co.uk"},
    "germany": {".de"},
    "australia": {".au", ".com.au"},
    "canada": {".ca"},
    "bangladesh": {".bd"},
    "china": {".cn"},
}

_COUNTRY_SERP_GL: dict[str, str] = {
    "pakistan": "pk",
    "india": "in",
    "singapore": "sg",
    "uae": "ae",
    "saudi arabia": "sa",
    "united states": "us",
    "united kingdom": "uk",
    "canada": "ca",
    "australia": "au",
    "germany": "de",
    "bangladesh": "bd",
}


def _normalize_country_key(market: str) -> str:
    text = _as_str(market).lower().strip()
    if not text:
        return ""
    # Prefer longest alias match so "united arab emirates" wins over "arab"
    best = ""
    best_len = 0
    for key, aliases in _COUNTRY_ALIASES.items():
        for alias in aliases:
            if alias in text and len(alias) > best_len:
                best = key
                best_len = len(alias)
        if key in text and len(key) > best_len:
            best = key
            best_len = len(key)
    return best


def _market_aliases(market: str) -> set[str]:
    key = _normalize_country_key(market)
    aliases = set(_COUNTRY_ALIASES.get(key, set()))
    raw = _as_str(market).lower().strip()
    if raw:
        aliases.add(raw)
        for tok in re.split(r"[^a-z0-9]+", raw):
            if len(tok) >= 3:
                aliases.add(tok)
    if key:
        aliases.add(key)
    return {a for a in aliases if a}


def _blob_mentions_any(blob: str, terms: set[str]) -> bool:
    if not blob or not terms:
        return False
    # Word-boundary-ish: prefer whole-word for short tokens (pk, sg, in, uk, ae)
    for term in terms:
        if len(term) <= 2:
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", blob):
                return True
        elif term in blob:
            return True
    return False


def _host_matches_tlds(host: str, tlds: set[str]) -> bool:
    if not host:
        return False
    for tld in tlds:
        suffix = tld[1:] if tld.startswith(".") else tld
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _mentions_target_market(blob: str, website: str | None, market: str) -> bool:
    aliases = _market_aliases(market)
    if _blob_mentions_any(blob, aliases):
        return True
    key = _normalize_country_key(market)
    host = _domain_of(website or "")
    if key and _host_matches_tlds(host, _COUNTRY_TLDS.get(key, set())):
        return True
    return False


def _mentions_conflicting_country(blob: str, website: str | None, market: str) -> bool:
    """True when text/site clearly points at a different known country than the required market."""
    target = _normalize_country_key(market)
    if not target:
        return False
    host = _domain_of(website or "")
    found: set[str] = set()
    for key, aliases in _COUNTRY_ALIASES.items():
        if key == target:
            continue
        if _blob_mentions_any(blob, aliases):
            found.add(key)
        if _host_matches_tlds(host, _COUNTRY_TLDS.get(key, set())):
            found.add(key)
    return bool(found)


def _serp_gl_for_market(market: str) -> str | None:
    key = _normalize_country_key(market)
    return _COUNTRY_SERP_GL.get(key)


# Peer-fit: reject consumer retail / media / wrong verticals when the client is a B2B software/agency peer
_B2B_PEER_MODELS = {"agency", "saas", "services", "product", "software", "consulting", "b2b"}
_RETAIL_MARKETPLACE_MARKERS = (
    "ecommerce", "e-commerce", "e commerce", "online shopping", "online store", "online retail",
    "shopping platform", "shopping mall", "marketplace", "cash on delivery", "cash-on-delivery",
    "fashion", "electronics store", "consumer durables", "grocery", "retail store", "retailer",
    "buy online", "add to cart", "shop now",
)
_MEDIA_DIRECTORY_MARKERS = (
    "tech news", "news portal", "blog", "magazine", "media company", "job board",
    "directory of", "review site", "listicle",
)
_GOVERNMENT_MARKERS = (
    "government", "govt", "gov.", "ministry", "public sector", "state-owned", "state owned",
    "federal board", "provincial board", "information technology board", "it board",
    "authority", "commission", "regulator", "municipal", "city government",
    "public body", "government of", "gov of", "pitb", "nadra", "fbr", "secp",
    "digital pakistan", "e-government", "egovernment", "smart city authority",
)
# Strong verticals — a rival dominated by one of these is NOT a peer unless the client shares it
_VERTICAL_MARKERS: dict[str, tuple[str, ...]] = {
    "fintech": (
        "fintech", "digital wallet", "mobile wallet", "e-wallet", "ewallet", "payment app",
        "payments", "payment gateway", "money transfer", "remittance", "neobank", "digital bank",
        "banking app", "lendtech", "buy now pay later", "bnpl", "credit card", "debit card",
        "wallet app", "send money", "cash in", "cash out", "iban", "branchless banking",
    ),
    "retail": _RETAIL_MARKETPLACE_MARKERS,
    "manufacturing": (
        "manufacturing", "pharmaceutical manufacturing", "process engineer", "digital twin",
        "plant optimization", "factory", "industrial automation",
    ),
    "telecom": ("telecom", "mobile network", "mobile operator", "isp ", "broadband provider", "5g network"),
    "healthcare": ("hospital", "clinic", "telemedicine", "healthcare provider", "pharma company", "medical device"),
    "edtech": ("edtech", "online learning", "e-learning", "school management", "university portal", "tutoring platform"),
    "logistics": ("logistics", "courier", "shipping company", "fleet management", "warehousing", "freight"),
    "real_estate": ("real estate", "property portal", "housing marketplace", "listings platform"),
    "government": _GOVERNMENT_MARKERS,
    "cybersecurity": ("cybersecurity", "endpoint security", "soc ", "threat detection", "penetration testing firm"),
    "data_ai": (
        "competitive intelligence", "market intelligence", "business intelligence", "data analytics",
        "ai agency", "machine learning platform", "data platform", "bi platform", "competitor tracking",
        "market research software", "insights platform",
    ),
    "software_services": (
        "software house", "software development company", "custom software", "it services",
        "digital agency", "web development agency", "product engineering", "dev shop",
        "digital engineering", "outsourcing software", "application development",
    ),
    "talent_marketplace": (
        "talent marketplace", "talent network", "hire developers", "staff augmentation marketplace",
        "freelance developers", "remote engineer marketplace", "vetting engineers",
        "andela", "turing.com", "toptal",
    ),
}
_MODEL_FAMILIES: dict[str, str] = {
    "agency": "b2b_services",
    "services": "b2b_services",
    "consulting": "b2b_services",
    "saas": "b2b_software",
    "product": "b2b_software",
    "software": "b2b_software",
    "b2b": "b2b_software",
    "marketplace": "marketplace",
    "ecommerce": "retail",
    "e-commerce": "retail",
    "retail": "retail",
    "shopping": "retail",
    "fintech": "fintech",
    "payments": "fintech",
    "other": "other",
}


def _model_family(value: str) -> str:
    raw = _as_str(value).lower().strip()
    if not raw:
        return ""
    if raw in _MODEL_FAMILIES:
        return _MODEL_FAMILIES[raw]
    for key, family in _MODEL_FAMILIES.items():
        if key in raw:
            return family
    return ""


def _detect_verticals(text: str) -> set[str]:
    blob = _as_str(text).lower()
    if not blob:
        return set()
    found: set[str] = set()
    for vertical, markers in _VERTICAL_MARKERS.items():
        hits = sum(1 for m in markers if m in blob)
        # fintech/retail need only one strong marker; others need a hit too
        if hits >= 1:
            found.add(vertical)
    # Name heuristics: NayaPay, EasyPaisa-style wallets are fintech even without long blurbs
    if re.search(r"\b\w{2,}pay\b", blob) or re.search(r"\b\w*wallet\b", blob) or "paisa" in blob:
        found.add("fintech")
    # PITB / IT boards / ministries
    if "pitb" in blob or (
        bool(re.search(r"\b\w+\s+board\b", blob))
        and any(tok in blob for tok in ("information technology", "it board", "government", "pakistan"))
    ):
        found.add("government")
    if ".gov." in blob or blob.endswith(".gov") or ".gob." in blob:
        found.add("government")
    return found


def _looks_like_government(blob: str) -> bool:
    text = _as_str(blob).lower()
    if not text:
        return False
    if "pitb" in text:
        return True
    return any(m in text for m in _GOVERNMENT_MARKERS)


_GENERIC_RIVAL_NAMES = {
    "techcorp", "tech corp", "tech-corp",
    "softcorp", "soft corp",
    "softsolutions", "soft solutions", "soft-solutions",
    "paktech", "pak tech", "paktech solutions", "pak tech solutions",
    "axonsoft", "axon soft",
    "techsoft", "tech soft",
    "infotech solutions", "info tech solutions",
    "global tech", "smart tech", "future tech", "nextgen tech", "next gen tech",
    "software solutions", "it solutions", "tech solutions", "digital solutions",
    "software house", "it company", "tech company", "software company",
    "abc tech", "xyz tech", "test company",
}
_GENERIC_NAME_RE = re.compile(
    r"^(tech|soft|pak|info|digital|global|smart|future|nextgen|next\s*gen|axon)"
    r"[\s\-]?(corp|soft|tech|solutions|systems|company|house)$",
    re.I,
)
_PARKED_SITE_MARKERS = (
    "domain for sale", "buy this domain", "this domain is for sale",
    "parked domain", "parkingcrew", "sedoparking", "godaddy parking",
    "coming soon", "under construction", "website coming soon",
    "account suspended", "default web page", "apache2 ubuntu default",
)
_SOFTWARE_PEER_SITE_MARKERS = (
    "software", "development", "digital agency", "web development", "mobile app",
    "it services", "custom software", "product engineering", "outsourcing",
    "app development", "devops", "saas", "solutions for",
)


def _is_generic_or_fake_rival_name(name: str) -> bool:
    """Block LLM placeholder brands like TechCorp / Soft Solutions / PakTech Solutions."""
    raw = _as_str(name).strip()
    if not raw:
        return True
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    compact = re.sub(r"[^a-z0-9]+", "", key)
    if key in _GENERIC_RIVAL_NAMES or compact in {re.sub(r"[^a-z0-9]+", "", n) for n in _GENERIC_RIVAL_NAMES}:
        return True
    if _GENERIC_NAME_RE.match(key):
        return True
    # Too short / too generic single-token brands
    if len(compact) < 5:
        return True
    # "X Solutions" with a very generic X
    if re.match(r"^(tech|soft|pak|it|info|digital|global|smart|web)\s+solutions$", key):
        return True
    return False


def _site_looks_parked_or_empty(site_md: str) -> bool:
    text = _as_str(site_md).lower().strip()
    if len(text) < 80:
        return True
    return any(m in text for m in _PARKED_SITE_MARKERS)


def _site_supports_software_peer(site_md: str) -> bool:
    text = _as_str(site_md).lower()
    if not text:
        return False
    return sum(1 for m in _SOFTWARE_PEER_SITE_MARKERS if m in text) >= 2


def _name_aligned_with_domain(name: str, website: str | None) -> bool:
    """Loose check: distinctive name token should appear in hostname when possible."""
    host = _domain_of(website or "")
    if not host:
        return False
    host_core = host.split(".")[0]
    tokens = [t for t in re.split(r"[^a-z0-9]+", _as_str(name).lower()) if len(t) >= 4]
    skip = {"solutions", "software", "technologies", "technology", "systems", "company", "limited", "private", "pakistan"}
    tokens = [t for t in tokens if t not in skip]
    if not tokens:
        return True  # can't judge
    return any(t in host_core or host_core in t for t in tokens)


# Curated fallbacks when SerpAPI is down — real commercial software houses only
_LOCAL_SOFTWARE_SEEDS: dict[str, list[dict]] = {
    "pakistan": [
        {"name": "Systems Limited", "website": "https://www.systemsltd.com"},
        {"name": "NetSol Technologies", "website": "https://www.netsoltech.com"},
        {"name": "10Pearls", "website": "https://10pearls.com"},
        {"name": "Arbisoft", "website": "https://arbisoft.com"},
        {"name": "Contour Software", "website": "https://www.contour-software.com"},
        {"name": "Folio3", "website": "https://www.folio3.com"},
        {"name": "Emumba", "website": "https://emumba.com"},
        {"name": "Confiz", "website": "https://www.confiz.com"},
        {"name": "VentureDive", "website": "https://www.venturedive.com"},
        {"name": "Tintash", "website": "https://www.tintash.com"},
        {"name": "Devsinc", "website": "https://www.devsinc.com"},
        {"name": "TekRevol", "website": "https://www.tekrevol.com"},
    ],
}
_GLOBAL_SOFTWARE_SEEDS: list[dict] = [
    {"name": "EPAM Systems", "website": "https://www.epam.com", "headquarters_country": "United States"},
    {"name": "Globant", "website": "https://www.globant.com", "headquarters_country": "Argentina"},
    {"name": "Endava", "website": "https://www.endava.com", "headquarters_country": "United Kingdom"},
    {"name": "Thoughtworks", "website": "https://www.thoughtworks.com", "headquarters_country": "United States"},
    {"name": "SoftServe", "website": "https://www.softserveinc.com", "headquarters_country": "United States"},
    {"name": "N-iX", "website": "https://www.n-ix.com", "headquarters_country": "Ukraine"},
    {"name": "Persistent Systems", "website": "https://www.persistent.com", "headquarters_country": "India"},
    {"name": "Intellias", "website": "https://www.intellias.com", "headquarters_country": "Ukraine"},
]


def _seed_local_software_rivals(
    market: str,
    client_name: str,
    *,
    already_have: list[str] | None = None,
    limit: int = 8,
) -> list[dict]:
    key = _normalize_country_key(market)
    seeds = _LOCAL_SOFTWARE_SEEDS.get(key) or []
    if not seeds:
        return []
    blocked = {_as_str(n).lower() for n in (already_have or [])}
    blocked.add(_as_str(client_name).lower())
    out: list[dict] = []
    for seed in seeds:
        name = _as_str(seed.get("name")).strip()
        if not name or name.lower() in blocked or _is_generic_or_fake_rival_name(name):
            continue
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not website:
            continue
        out.append(
            {
                "name": name,
                "website": website,
                "industry": "Software",
                "business_model": "services",
                "headquarters_country": key.title() if key else market,
                "why_relevant": (
                    f"Established commercial software house / digital product firm in {market}; "
                    f"peer IT services rival for local software buyers."
                ),
                "threat_level": "high",
                "overlap_score": 72,
                "same_niche": True,
                "same_market": True,
                "source": "seed",
            }
        )
        if len(out) >= limit:
            break
    return out


def _seed_global_software_rivals(
    client_name: str,
    *,
    already_have: list[str] | None = None,
    limit: int = 8,
) -> list[dict]:
    blocked = {_as_str(n).lower() for n in (already_have or [])}
    blocked.add(_as_str(client_name).lower())
    out: list[dict] = []
    for seed in _GLOBAL_SOFTWARE_SEEDS:
        name = _as_str(seed.get("name")).strip()
        if not name or name.lower() in blocked or _is_generic_or_fake_rival_name(name):
            continue
        website = _normalize_website(_as_str(seed.get("website")) or None)
        if not website:
            continue
        out.append(
            {
                "name": name,
                "website": website,
                "industry": "Software",
                "business_model": "services",
                "headquarters_country": _as_str(seed.get("headquarters_country")) or "Global",
                "why_relevant": (
                    "Global custom software / digital engineering firm competing for similar enterprise buyers."
                ),
                "threat_level": "high",
                "overlap_score": 70,
                "same_niche": True,
                "same_market": True,
                "source": "seed",
            }
        )
        if len(out) >= limit:
            break
    return out


def _is_curated_seed_rival(name: str, market: str | None = None) -> bool:
    key = _as_str(name).lower().strip()
    if not key:
        return False
    for seed in _GLOBAL_SOFTWARE_SEEDS:
        if _as_str(seed.get("name")).lower() == key:
            return True
    market_key = _normalize_country_key(market or "")
    for country_key, seeds in _LOCAL_SOFTWARE_SEEDS.items():
        if market_key and country_key != market_key:
            continue
        for seed in seeds:
            if _as_str(seed.get("name")).lower() == key:
                return True
    return False


def _looks_like_retail_or_media(blob: str) -> str | None:
    """Return 'retail' or 'media' when blob clearly isn't a B2B peer company."""
    text = _as_str(blob).lower()
    if not text:
        return None
    retail_hits = sum(1 for m in _RETAIL_MARKETPLACE_MARKERS if m in text)
    if retail_hits >= 1 and any(
        m in text
        for m in ("shop", "shopping", "retail", "marketplace", "ecommerce", "e-commerce", "store", "cart")
    ):
        return "retail"
    if any(m in text for m in _MEDIA_DIRECTORY_MARKERS):
        return "media"
    return None


def _incompatible_peer(
    *,
    client_model: str,
    client_industry: str,
    client_niche: str,
    rival_model: str,
    rival_industry: str,
    rival_blob: str,
) -> bool:
    """True when rival is clearly not the same kind of business as the client."""
    client_family = _model_family(client_model)
    rival_family = _model_family(rival_model)
    client_l = f"{client_model} {client_industry} {client_niche}".lower()
    rival_l = f"{rival_model} {rival_industry} {rival_blob}".lower()
    client_is_b2b = client_family in {"b2b_services", "b2b_software"} or any(
        tok in client_l for tok in _B2B_PEER_MODELS
    ) or any(
        tok in client_l
        for tok in ("ai", "software", "saas", "agency", "data", "analytics", "intelligence", "automation", "technology")
    )

    client_verticals = _detect_verticals(client_l)
    rival_verticals = _detect_verticals(rival_l)
    # Strong alternate verticals that should not match a generic "Technology" / AI / agency client
    hard_verticals = {
        "fintech", "retail", "manufacturing", "telecom", "healthcare",
        "edtech", "logistics", "real_estate", "government", "talent_marketplace",
    }
    peer_verticals = {"data_ai", "software_services", "cybersecurity"}

    if rival_verticals & hard_verticals:
        # Client must share that vertical (or explicitly be in it)
        if not (client_verticals & rival_verticals & hard_verticals):
            # Exception: only if client is also tagged with that vertical in industry/niche
            return True

    # Software houses / digital agencies are not peers of talent marketplaces (Andela/Turing/Toptal)
    if "talent_marketplace" in rival_verticals and "talent_marketplace" not in client_verticals:
        return True
    if any(tok in rival_l for tok in ("andela", "turing", "toptal")) and "talent_marketplace" not in client_verticals:
        if any(tok in client_l for tok in ("software", "agency", "development", "digital", "it services")):
            return True

    # Commercial software houses / agencies never compete with government boards/authorities
    client_is_commercial_software = any(
        tok in client_l
        for tok in (
            "software house", "software", "agency", "saas", "services", "it services",
            "digital agency", "technology", "product",
        )
    ) and "government" not in client_verticals
    if client_is_commercial_software and (
        "government" in rival_verticals or _looks_like_government(rival_l)
    ):
        return True

    if client_is_b2b:
        kind = _looks_like_retail_or_media(rival_blob)
        if kind in {"retail", "media"}:
            return True
        if rival_family in {"retail", "fintech"} and "fintech" not in client_verticals and "retail" not in client_verticals:
            return True
        # Manufacturing / industrial plant AI is not a peer for marketing/data agencies
        if "manufacturing" in rival_verticals and "manufacturing" not in client_verticals:
            return True
        # If client looks like data/AI/software services, rival must not be pure fintech/payments
        if (client_verticals & peer_verticals or any(
            tok in client_l for tok in ("ai", "data", "analytics", "intelligence", "agency", "software", "saas")
        )) and ("fintech" in rival_verticals) and ("fintech" not in client_verticals):
            return True

    if client_family and rival_family and client_family != rival_family:
        if {client_family, rival_family} == {"b2b_services", "b2b_software"}:
            return False  # agency vs saas can still be peers in some niches
        if {"retail", "marketplace", "fintech"} & {client_family, rival_family}:
            return True
    return False


def _is_global_megarival(name: str, website: str | None = None) -> bool:
    n = _as_str(name).strip().lower()
    if not n:
        return False
    if n in _GLOBAL_RIVAL_BLOCKLIST:
        return True
    for blocked in _GLOBAL_RIVAL_BLOCKLIST:
        if len(blocked) < 4:
            continue
        if blocked == n or n.startswith(blocked + " ") or n.endswith(" " + blocked) or f" {blocked} " in f" {n} ":
            return True
        if blocked in n and blocked not in {"ai", "aws", "ibm", "sap", "ey", "tcs", "bcg"}:
            return True
    host = _domain_of(website or "")
    if host:
        for blocked in _GLOBAL_DOMAIN_BLOCKLIST:
            if host == blocked or host.endswith("." + blocked):
                return True
    return False


def _market_area_from_client(client: ClientBrand) -> str:
    notes = _as_str(client.notes)
    for line in notes.splitlines():
        if line.lower().startswith("market:"):
            return line.split(":", 1)[1].strip()
    return ""


def _business_model_from_client(client: ClientBrand) -> str:
    notes = _as_str(client.notes)
    for line in notes.splitlines():
        if line.lower().startswith("business model:"):
            return line.split(":", 1)[1].strip()
    return ""


def _set_market_area(client: ClientBrand, market_area: str) -> None:
    market_area = _as_str(market_area).strip()
    notes = _as_str(client.notes)
    lines = [ln for ln in notes.splitlines() if not ln.lower().startswith("market:")]
    if market_area:
        lines.insert(0, f"Market: {market_area}")
    client.notes = "\n".join(lines).strip() or None


def _set_business_model(client: ClientBrand, business_model: str) -> None:
    business_model = _as_str(business_model).strip()
    notes = _as_str(client.notes)
    lines = [ln for ln in notes.splitlines() if not ln.lower().startswith("business model:")]
    if business_model:
        # Keep Market: first when present
        insert_at = 1 if lines and lines[0].lower().startswith("market:") else 0
        lines.insert(insert_at, f"Business model: {business_model}")
    client.notes = "\n".join(lines).strip() or None


def _niche_competitor_queries(client: ClientBrand, market_area: str = "") -> list[str]:
    niche = _as_str(client.niche) or _as_str(client.industry) or "software"
    market = market_area or _market_area_from_client(client)
    model = _business_model_from_client(client).lower()
    niche_l = niche.lower()
    queries = [
        f"{client.name} competitors {niche}",
        f"companies like {client.name} {niche}",
        f"{niche} agencies competitors {client.name}",
    ]
    # Software-house / IT services clients need peer-shaped queries, not generic "technology"
    if any(tok in f"{niche_l} {model} {_as_str(client.industry).lower()}" for tok in (
        "software", "agency", "it services", "development", "digital",
    )):
        if market:
            queries.extend(
                [
                    f"top software houses in {market}",
                    f"software development companies in {market}",
                    f"digital agencies {market} like {client.name}",
                    f"IT services companies {market}",
                    f"{client.name} competitors software house {market}",
                ]
            )
        else:
            queries.extend(
                [
                    f"software development companies like {client.name}",
                    f"digital agencies competitors {client.name}",
                ]
            )
    if market:
        queries.extend(
            [
                f"{niche} companies in {market}",
                f"{niche} agencies {market} like {client.name}",
                f"{client.name} competitors {market}",
            ]
        )
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        key = q.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(q.strip())
    return out[:8]


_SERP_NOISE_DOMAINS = {
    "g2.com", "capterra.com", "getapp.com", "softwareadvice.com", "trustradius.com",
    "clutch.co", "goodfirms.co", "sortlist.com", "designrush.com", "upcity.com",
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
    "wikipedia.org", "crunchbase.com", "bloomberg.com", "forbes.com", "techcrunch.com",
    "medium.com", "reddit.com", "quora.com", "glassdoor.com", "indeed.com",
    "producthunt.com", "alternativeto.net", "saashub.com", "slashdot.org",
}


def _is_serp_noise_domain(url: str) -> bool:
    host = _domain_of(url)
    if not host:
        return True
    for blocked in _SERP_NOISE_DOMAINS:
        if host == blocked or host.endswith("." + blocked):
            return True
    return False


def _token_hits(haystack: str, source: str, *, min_len: int = 3) -> int:
    if not haystack or not source:
        return 0
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", source.lower()) if len(tok) >= min_len]
    if not tokens:
        return 0
    return sum(1 for tok in tokens if tok in haystack)


def _filter_niche_competitors(
    items: list[dict],
    client_name: str,
    *,
    market_area: str = "",
    niche: str = "",
    industry: str = "",
    business_model: str = "",
    min_overlap: float = 55.0,
    limit: int = 10,
    require_local_market: bool = False,
) -> list[dict]:
    """Keep only peer rivals that fit niche/industry; rank by relevance score."""
    scored: list[dict] = []
    seen_names: set[str] = set()
    seen_hosts: set[str] = set()
    client_l = client_name.lower().strip()
    niche_l = niche.lower().strip()
    industry_l = industry.lower().strip()
    market_l = market_area.lower().strip()
    model_l = business_model.lower().strip()

    for item in items:
        if not isinstance(item, dict):
            continue
        name = _as_str(item.get("name")).strip()
        website = _normalize_website(_as_str(item.get("website")) or None)
        if not name or name.lower() == client_l or name.lower() in seen_names:
            continue
        if _is_generic_or_fake_rival_name(name):
            continue
        host = _domain_of(website or "")
        if host and host in seen_hosts:
            continue
        if _is_global_megarival(name, website):
            continue
        if website and _is_serp_noise_domain(website):
            continue
        if item.get("same_niche") is False or item.get("is_global_platform") is True:
            continue
        if require_local_market and item.get("same_market") is False:
            continue
        # Invented AI rivals often ship a website that doesn't match the brand
        alignment_ok = (
            (not website)
            or _as_str(item.get("source")).lower() in {"serp", "seed"}
            or _name_aligned_with_domain(name, website)
        )
        if website and not alignment_ok and require_local_market:
            continue

        why = _as_str(item.get("why_relevant") or item.get("description"))
        item_industry = _as_str(item.get("industry"))
        item_model = _as_str(item.get("business_model"))
        hq_country = _as_str(item.get("headquarters_country") or item.get("headquarters") or item.get("market_overlap"))
        try:
            score = float(item.get("overlap_score") or item.get("niche_fit_score") or 50)
        except (TypeError, ValueError):
            score = 50.0
        if not alignment_ok:
            score -= 20

        blob = f"{name} {why} {website or ''} {item_industry} {item_model} {hq_country}".lower()

        if _incompatible_peer(
            client_model=model_l,
            client_industry=industry_l,
            client_niche=niche_l,
            rival_model=item_model,
            rival_industry=item_industry,
            rival_blob=blob,
        ):
            continue

        # Local scope: hard-reject clear foreign-country rivals (e.g. India/Singapore when market=Pakistan)
        if require_local_market and market_l:
            if _mentions_conflicting_country(blob, website, market_l):
                continue
            hq_key = _normalize_country_key(hq_country)
            market_key = _normalize_country_key(market_l)
            if hq_key and market_key and hq_key != market_key:
                continue
            has_local_signal = _mentions_target_market(blob, website, market_l)
            if not has_local_signal:
                # SERP rows are provisional — pack scrape verifies HQ later
                if _as_str(item.get("source")).lower() == "serp":
                    score -= 8
                else:
                    # AI must cite the country/city; bare same_market=true is not enough
                    continue

        # Soft boosts for explicit fit signals
        if item.get("same_niche") is True:
            score += 12
        if niche_l:
            hits = _token_hits(blob, niche_l)
            if hits:
                score += min(14, hits * 5)
            elif len([t for t in niche_l.replace("/", " ").split() if len(t) > 3]) >= 1:
                # No niche token overlap → penalize vague AI guesses
                score -= 12
        if industry_l:
            hits = _token_hits(blob, industry_l) + _token_hits(item_industry.lower(), industry_l)
            if hits:
                score += min(12, hits * 4)
            else:
                score -= 8
        if model_l:
            hits = _token_hits(blob, model_l) + _token_hits(item_model.lower(), model_l)
            if hits:
                score += min(10, hits * 4)
        if market_l:
            if _mentions_target_market(blob, website, market_l):
                score += 16
            elif require_local_market:
                score -= 30

        # Local runs need a usable website when one is claimed
        if require_local_market and _as_str(item.get("website")) and not website:
            continue

        score = max(0.0, min(score, 95.0))
        local_min = max(min_overlap, 60.0) if require_local_market else min_overlap
        if score < local_min:
            continue

        seen_names.add(name.lower())
        if host:
            seen_hosts.add(host)
        scored.append(
            {
                **item,
                "name": name,
                "website": website,
                "industry": item_industry or item.get("industry"),
                "business_model": item_model or item.get("business_model"),
                "why_relevant": why or item.get("why_relevant"),
                "overlap_score": score,
                "threat_level": _as_str(item.get("threat_level"), "high").lower(),
            }
        )

    scored.sort(key=lambda row: float(row.get("overlap_score") or 0), reverse=True)
    return scored[: max(1, limit)]


def _competitors_from_serp(organic: list[dict], client_name: str) -> list[dict]:
    rivals: list[dict] = []
    seen: set[str] = set()
    client_l = client_name.lower()
    skip_title_bits = (
        "vs ", " versus ", "alternative", "alternatives", "best ", "top ", "compared",
        "review", "pricing", "jobs", "career", "salary", "news", "blog",
    )
    for item in organic or []:
        title = _as_str(item.get("title"))
        link = _as_str(item.get("link"))
        snippet = _as_str(item.get("snippet"))
        if not title or not link:
            continue
        if _is_serp_noise_domain(link):
            continue
        name = title.split("|")[0].split("-")[0].split("–")[0].strip()
        if not name or client_l in name.lower() or len(name) > 60:
            continue
        if _is_generic_or_fake_rival_name(name):
            continue
        lowered = name.lower()
        if any(bit in lowered for bit in skip_title_bits):
            continue
        # Skip listicle-style titles that aren't company names
        if lowered.startswith(("the ", "how ", "what ", "why ", "10 ", "5 ", "7 ", "15 ")):
            continue
        if _is_global_megarival(name, link):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        rivals.append(
            {
                "name": name,
                "website": _normalize_website(link.split("?")[0]),
                "why_relevant": snippet[:220] or f"Appears in niche search for {client_name} competitors",
                "threat_level": "medium",
                # Conservative until enrich validates niche fit
                "overlap_score": 52,
                "same_niche": True,
                "source": "serp",
            }
        )
        if len(rivals) >= 10:
            break
    return rivals



async def enrich_client_profile(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    competitor_scope: str = "local",
    competitor_country: str | None = None,
    competitor_count: int = 5,
    competitor_mode: str = "add",
) -> dict:
    scope = "global" if str(competitor_scope).lower() == "global" else "local"
    raw_mode = str(competitor_mode or "add").strip().lower()
    mode = raw_mode if raw_mode in {"update", "add", "replace"} else "add"
    count = max(1, min(10, int(competitor_count or 5)))
    country = _as_str(competitor_country).strip()

    site = {}
    if client.website:
        site = await scrape_website(db, agency.id, client.website)
    site_md = (site.get("markdown") or "")[:4500]

    profile = await ai_service.structured_json(
        db,
        agency.id,
        (
            "Profile this company for competitive intelligence. "
            "Return JSON keys ONLY: industry, niche, market_area, business_model, tagline, description, "
            "goals (3-5 strings), features (8-12 objects with name, category, description). "
            "niche = specific category (not just 'AI' or 'Software'). "
            "market_area = concrete city/region/country they sell into "
            "(e.g. 'Pakistan', 'Karachi', 'UAE', 'MENA', 'US mid-market'). "
            "Never return only 'Global' or 'Worldwide' — use HQ or primary selling region from contact/address/phone clues. "
            "business_model = agency|product|saas|services|marketplace|other. "
            "For each feature.description: write 2–3 plain-English sentences a non-technical person can understand. "
            "Explain what the customer gets and why it matters. No slogans, no unexplained jargon "
            "(avoid 'production-grade', 'demoware', 'architecture-first' unless explained simply). "
            "Use the website excerpt. Be concrete. No filler."
        ),
        json.dumps(
            {
                "name": client.name,
                "website": client.website,
                "industry_hint": client.industry,
                "site_markdown": site_md,
                "preferred_market": country or None,
            }
        )[:7000],
        temperature=0.2,
    )

    if not isinstance(profile.get("features"), list) or not profile.get("features"):
        profile = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Return JSON: {industry, niche, market_area, business_model, tagline, description, "
                "goals:[], features:[{name, category, description}]}."
            ),
            json.dumps(
                {
                    "name": client.name,
                    "website": client.website,
                    "excerpt": site_md[:2500],
                }
            ),
            temperature=0.15,
        )

    client.industry = _as_str(profile.get("industry")) or client.industry or "Software"
    client.niche = _as_str(profile.get("niche")) or client.niche
    client.tagline = _as_str(profile.get("tagline")) or client.tagline
    market_area = _as_str(profile.get("market_area")) or _market_area_from_client(client)
    if market_area.strip().lower() in {"global", "worldwide", "international", "world"}:
        market_area = ""
    if scope == "local" and country:
        market_area = country
    business_model = _as_str(profile.get("business_model")) or _business_model_from_client(client) or "services"
    description = _as_str(profile.get("description"))
    if description:
        client.notes = description
    _set_market_area(client, market_area)
    _set_business_model(client, business_model)

    goals = profile.get("goals") if isinstance(profile.get("goals"), list) else []
    client.goals = [_as_str(g) for g in goals if _as_str(g)] or client.goals or [
        "Win more competitive deals",
        "Close product gaps vs leading rivals",
        "Improve category positioning",
    ]

    feature_items = profile.get("features") if isinstance(profile.get("features"), list) else []
    if not feature_items:
        feature_items = _extract_features_from_markdown(site_md)
    if not feature_items:
        industry_hint = _as_str(client.industry) or _as_str(profile.get("industry")) or "this category"
        feature_items = [
            {
                "name": f"Core {industry_hint} offering",
                "category": "Product",
                "description": f"Primary products or services this brand sells in {industry_hint}.",
            },
            {
                "name": "Customer onboarding",
                "category": "Experience",
                "description": "How new customers get started and reach first value.",
            },
            {
                "name": "Delivery / implementation",
                "category": "Services",
                "description": "How the company delivers work, support, or product updates.",
            },
            {
                "name": "Pricing & packaging",
                "category": "Commercial",
                "description": "Plans, packages, or engagement models sold to buyers.",
            },
            {
                "name": "Proof & credibility",
                "category": "Marketing",
                "description": "Case studies, testimonials, certifications, or public proof points.",
            },
        ]

    existing_features = (
        await db.execute(
            select(ProductFeature).where(ProductFeature.client_id == client.id, ProductFeature.agency_id == agency.id)
        )
    ).scalars().all()
    features_by_name = {_as_str(f.name).lower(): f for f in existing_features}
    feature_rows: list[ProductFeature] = list(existing_features)
    for item in feature_items[:14]:
        if not isinstance(item, dict):
            name = _as_str(item).strip()
            item = {"name": name, "category": "General", "description": name}
        name = _as_str(item.get("name"), "Feature").strip()
        if not name:
            continue
        key = name.lower()
        if key in features_by_name:
            feat = features_by_name[key]
            feat.category = _as_str(item.get("category") or feat.category, "General")
            if item.get("description"):
                feat.description = _as_str(item.get("description"))
        else:
            feature = ProductFeature(
                agency_id=agency.id,
                client_id=client.id,
                name=name,
                category=_as_str(item.get("category"), "General"),
                description=_as_str(item.get("description")),
            )
            db.add(feature)
            features_by_name[key] = feature
            feature_rows.append(feature)

    await db.flush()
    await clarify_feature_descriptions(db, agency, client, feature_rows)

    existing_early = (
        await db.execute(select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id))
    ).scalars().all()
    # replace: drop auto-found rivals (keep pinned/manual), then discover a fresh set
    if mode == "replace":
        for competitor in existing_early:
            if not competitor.is_pinned:
                competitor.is_tracking = False
        await db.flush()
    tracking_existing_early = (
        [c for c in existing_early if c.is_pinned]
        if mode == "replace"
        else [c for c in existing_early if c.is_tracking or c.is_pinned]
    )
    # replace: only avoid pinned/manual names — previously auto-found rivals may be reselected
    already_have_names = (
        [_as_str(c.name) for c in existing_early if c.is_pinned and _as_str(c.name)]
        if mode == "replace"
        else [_as_str(c.name) for c in tracking_existing_early]
    )
    if mode == "update":
        await db.flush()
        return {
            "features": len(feature_rows),
            "competitors_added": 0,
            "competitors_requested": count,
            "competitor_scope": scope,
            "competitor_country": country or None,
            "competitors_kept_existing": len(tracking_existing_early),
            "competitors_pruned_global": 0,
            "competitor_mode": mode,
            "goals": len(client.goals or []),
            "industry": client.industry,
            "niche": client.niche,
            "market_area": market_area,
            "business_model": business_model,
        }

    if scope == "global":
        competitor_prompt = (
            f"Find exactly {count} REAL direct competitors for this company with GLOBAL / international reach. "
            "They must compete in the SAME niche, SAME industry, and similar business model / buyer. "
            f"Must-match industry: {_as_str(client.industry) or 'unknown'}. "
            f"Must-match niche: {_as_str(client.niche) or 'unknown'}. "
            f"Must-match business model: {_as_str(business_model) or 'unknown'}. "
            f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, why_relevant, threat_level, overlap_score, "
            "same_niche:true, same_market:true, market_overlap, is_global_platform:false{'}'}]}}. "
            "Hard rules:\n"
            f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
            "2) Only peer businesses selling a similar product/service to similar buyers — not adjacent tools.\n"
            "3) EXCLUDE directories, review sites, job boards, news articles, and hyperscaler platforms "
            "(AWS/Azure/GCP as clouds, Dialogflow as a raw API) unless they are a true peer product.\n"
            "4) why_relevant must cite industry + niche + buyer overlap.\n"
            "5) overlap_score should reflect true peer fit (prefer 60-95). Reject weak/tangential names.\n"
            "6) Only include companies you believe actually exist with real websites. "
            "Never invent placeholder brands like TechCorp, Soft Solutions, PakTech Solutions, AxonSoft."
        )
    else:
        focus = country or market_area or "the client's primary country/region"
        competitor_prompt = (
            f"Find exactly {count} REAL direct LOCAL / country competitors for this company in {focus}. "
            f"HARD GEO RULE: every competitor MUST be headquartered in OR primarily selling in {focus}. "
            f"Do NOT return companies from other countries (e.g. if focus is Pakistan, exclude India, Singapore, UAE, US, UK rivals). "
            "They must be from the SAME niche, SAME industry, SAME business model, and the SAME country/market. "
            f"Must-match industry: {_as_str(client.industry) or 'unknown'}. "
            f"Must-match niche: {_as_str(client.niche) or 'unknown'}. "
            f"Must-match business model: {_as_str(business_model) or 'unknown'}. "
            f"Return JSON: {{competitors:[{'{'}name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, "
            "same_niche:true, same_market:true, market_overlap, is_global_platform:false{'}'}]}}. "
            "Hard rules:\n"
            f"1) Return exactly {count} NEW competitors — not names in already_have.\n"
            f"2) headquarters_country MUST be {focus} (or a city inside {focus}).\n"
            f"3) why_relevant MUST mention {focus} and how they sell there.\n"
            f"4) website MUST be a real working company homepage URL (https://...). No invented domains.\n"
            "5) EXCLUDE consumer shopping / ecommerce retailers / marketplaces "
            "(Daraz, Telemart, Amazon-style stores) unless the client itself is retail ecommerce.\n"
            "6) EXCLUDE fintech wallets, payment apps, banks, and remittance apps "
            "(NayaPay, EasyPaisa, JazzCash, SadaPay) unless the client itself is fintech/payments.\n"
            "7) EXCLUDE government boards, ministries, regulators, and public-sector IT bodies "
            "(PITB, NADRA, ministries, authorities) — they are not commercial software-house rivals.\n"
            "8) EXCLUDE global hyperscalers and mega consultancies "
            "(Accenture, IBM, Microsoft, Google, Amazon/AWS, Oracle, SAP, Deloitte, PwC, EY, KPMG, Cognizant, Infosys, TCS, Wipro, OpenAI).\n"
            "9) EXCLUDE directories, review sites, and tools/infrastructure that are not peer businesses.\n"
            f"10) If you are unsure a company is based in / sells primarily in {focus}, OMIT it.\n"
            "11) Same 'Technology' industry is NOT enough — they must sell a similar product/service to similar buyers.\n"
            "12) Prefer commercial software houses / digital agencies / IT services firms as peers for a software house client.\n"
            "13) NEVER invent placeholder brands (TechCorp, Soft Solutions, SoftCorp, PakTech Solutions, AxonSoft, IT Solutions). "
            "Only well-known or clearly real companies with working websites.\n"
            "14) overlap_score should reflect true peer fit (prefer 60-95).\n"
            "15) Only include companies you believe actually exist."
        )

    def _apply_relevance_filter(rows: list[dict]) -> list[dict]:
        local_market = (country or market_area) if scope == "local" else ""
        return _filter_niche_competitors(
            rows,
            client.name,
            market_area=local_market,
            niche=_as_str(client.niche),
            industry=_as_str(client.industry),
            business_model=_as_str(business_model),
            min_overlap=55.0,
            limit=max(count * 2, 10),
            require_local_market=(scope == "local"),
        )

    competitor_items: list[dict] = []
    local_focus = country or market_area or ""
    serp_auth_failed = False

    # SERP-first for local runs — reduces LLM-invented brands like TechCorp / Soft Solutions
    serp_budget = 6 if scope == "local" else 3
    for query in _niche_competitor_queries(client, local_focus if scope == "local" else (country or market_area or ""))[:serp_budget]:
        if scope == "local" and local_focus and local_focus.lower() not in query.lower():
            query = f"{query} {local_focus}"
        elif scope == "global":
            query = f"{query} global competitors"
        serp = await serp_visibility(
            db,
            agency.id,
            query,
            location=local_focus if scope == "local" else None,
            gl=_serp_gl_for_market(local_focus) if scope == "local" else None,
        )
        status = _as_str(serp.get("status")).lower()
        detail_l = _as_str(serp.get("detail")).lower()
        if status == "unauthorized" or "unauthorized" in detail_l or "invalid api key" in detail_l:
            serp_auth_failed = True
        competitor_items.extend(_competitors_from_serp(serp.get("organic") or [], client.name))
        competitor_items = _apply_relevance_filter(competitor_items)
        if len(competitor_items) >= max(count, 4):
            break

    # When SerpAPI is broken/empty, seed real software-house peers so intel still works
    client_blob = f"{client.industry} {client.niche} {business_model}".lower()
    is_software_peer_client = any(
        tok in client_blob for tok in ("software", "agency", "technology", "ai", "it ", "digital", "development")
    )
    if is_software_peer_client and len(competitor_items) < max(2, count // 2):
        already = already_have_names + [_as_str(c.get("name")) for c in competitor_items]
        if scope == "local" and local_focus:
            competitor_items.extend(
                _seed_local_software_rivals(
                    local_focus,
                    client.name,
                    already_have=already,
                    limit=max(count * 2, 8),
                )
            )
        else:
            competitor_items.extend(
                _seed_global_software_rivals(
                    client.name,
                    already_have=already,
                    limit=max(count * 2, 8),
                )
            )
        competitor_items = _apply_relevance_filter(competitor_items)
        if serp_auth_failed:
            logger.warning(
                "SerpAPI unauthorized for agency=%s — using curated %s software-house seeds",
                agency.id,
                local_focus if scope == "local" else "global",
            )

    # AI ranks/fills — prefer choosing from SERP candidates when available
    serp_names = [_as_str(c.get("name")) for c in competitor_items if _as_str(c.get("name"))]
    ai_payload = {
        "name": client.name,
        "website": client.website,
        "industry": client.industry,
        "niche": client.niche,
        "market_area": market_area,
        "competitor_scope": scope,
        "competitor_country": country or None,
        "competitor_count": count,
        "competitor_mode": mode,
        "already_have": already_have_names,
        "business_model": business_model,
        "features": [f.name for f in feature_rows[:10]],
        "site_excerpt": site_md[:2000],
        "serp_candidates": competitor_items[:12],
    }
    if serp_names:
        competitor_prompt = (
            competitor_prompt
            + "\n15) Prefer picking from serp_candidates when they are true peers. "
            "You may add other REAL peers only if serp_candidates are insufficient — never invent placeholder names."
        )
    competitor_pack = await ai_service.structured_json(
        db,
        agency.id,
        competitor_prompt,
        json.dumps(ai_payload)[:9000],
        temperature=0.15,
    )
    if isinstance(competitor_pack.get("competitors"), list):
        competitor_items.extend([c for c in competitor_pack["competitors"] if isinstance(c, dict)])
    competitor_items = _apply_relevance_filter(competitor_items)

    # Prefer keeping at least `count` candidates before AI can shrink the pool
    if is_software_peer_client and len(competitor_items) < count:
        already = already_have_names + [_as_str(c.get("name")) for c in competitor_items]
        if scope == "local" and local_focus:
            competitor_items.extend(
                _seed_local_software_rivals(
                    local_focus,
                    client.name,
                    already_have=already,
                    limit=max(count * 2, 8),
                )
            )
        else:
            competitor_items.extend(
                _seed_global_software_rivals(
                    client.name,
                    already_have=already,
                    limit=max(count * 2, 8),
                )
            )
        competitor_items = _apply_relevance_filter(competitor_items)

    min_needed = max(1, min(count, 4))
    if len(competitor_items) < min_needed:
        retry_pack = await ai_service.structured_json(
            db,
            agency.id,
            (
                f"Propose exactly {count} REAL niche peer competitors "
                + (
                    f"headquartered in {country or market_area or 'the local market'} only. "
                    f"Exclude any company based in a different country. "
                    f"Each must include headquarters_country={country or market_area} and a real https website. "
                    "Do NOT invent TechCorp/Soft Solutions/PakTech-style placeholder names."
                    if scope == "local"
                    else "with global/international reach. Do not invent placeholder brands."
                )
                + " They MUST match the client's industry, niche, and business model. "
                + "Return JSON {competitors:[{name, website, industry, business_model, headquarters_country, why_relevant, threat_level, overlap_score, same_niche:true, same_market:true}]}. "
                + (
                    "No Fortune-500 mega-platforms. Only real local software houses / digital agencies / IT services firms."
                    if scope == "local"
                    else "Prefer known international category peers."
                )
            ),
            json.dumps(
                {
                    "name": client.name,
                    "niche": client.niche,
                    "industry": client.industry,
                    "market_area": market_area,
                    "competitor_scope": scope,
                    "competitor_country": country or None,
                    "competitor_count": count,
                    "business_model": business_model,
                    "already_have": [c.get("name") for c in competitor_items] + already_have_names,
                    "serp_candidates": serp_names[:12],
                }
            ),
            temperature=0.2,
        )
        if isinstance(retry_pack.get("competitors"), list):
            competitor_items.extend([c for c in retry_pack["competitors"] if isinstance(c, dict)])
        competitor_items = _apply_relevance_filter(competitor_items)

    existing = (
        await db.execute(select(Competitor).where(Competitor.client_id == client.id, Competitor.agency_id == agency.id))
    ).scalars().all()
    by_name = {_as_str(c.name).lower(): c for c in existing}

    # Keep existing/manual rivals — boost + pin so they survive AI prune & count slices.
    # replace: only pinned/manual stay; auto-found were already untracked above.
    protected_existing = (
        [c for c in existing if c.is_pinned]
        if mode == "replace"
        else [c for c in existing if c.is_tracking or c.is_pinned]
    )
    for competitor in protected_existing:
        looks_manual = (competitor.overlap_score or 0) < 55 or not (competitor.feature_list or [])
        competitor.is_tracking = True
        if (competitor.overlap_score or 0) < 55:
            competitor.overlap_score = 70.0
        if not competitor.threat_level or competitor.threat_level == "low":
            competitor.threat_level = "medium"
        if looks_manual or competitor.is_pinned:
            competitor.is_pinned = True
        if not competitor.why_dangerous:
            competitor.why_dangerous = f"Existing rival kept for {client.name}"

    pruned_global = 0
    for competitor in existing:
        # Never auto-drop pinned (manual) rivals
        if competitor.is_pinned:
            continue
        if mode == "replace":
            competitor.is_tracking = False
            continue
        if _is_global_megarival(competitor.name, competitor.website):
            competitor.is_tracking = False
            competitor.threat_level = "low"
            competitor.overlap_score = min(float(competitor.overlap_score or 0), 30)
            pruned_global += 1

    # add: find exactly `count` NEW rivals on top of previous ones.
    # replace: rebuild up to `count` auto rivals (may re-enable previously untracked rows; keep pinned).
    ai_slots = count
    existing_names = (
        {_as_str(c.name).lower() for c in existing if c.is_pinned and _as_str(c.name)}
        if mode == "replace"
        else {_as_str(c.name).lower() for c in protected_existing}
    )
    fresh_items = []
    seen_fresh: set[str] = set()
    for c in competitor_items:
        if not isinstance(c, dict):
            continue
        key = _as_str(c.get("name")).lower()
        if not key or key in existing_names or key in seen_fresh:
            continue
        if _is_generic_or_fake_rival_name(_as_str(c.get("name"))):
            continue
        if not _normalize_website(_as_str(c.get("website")) or None):
            continue
        seen_fresh.add(key)
        fresh_items.append(c)
        if len(fresh_items) >= ai_slots:
            break
    deduped = fresh_items

    created_competitors = 0
    for item in deduped:
        name = _as_str(item.get("name")).strip()
        if _is_generic_or_fake_rival_name(name):
            continue
        threat = _as_str(item.get("threat_level"), "high").lower()
        try:
            overlap = float(item.get("overlap_score") or 70)
        except (TypeError, ValueError):
            overlap = 70.0
        key = name.lower()
        why = _as_str(item.get("why_relevant"))
        if not name:
            continue
        website = _normalize_website(_as_str(item.get("website")) or None)
        if not website:
            continue
        if key in by_name:
            # replace mode skips previously known names via existing_names; this path is for add/re-enable
            competitor = by_name[key]
            competitor.website = website or competitor.website
            competitor.description = why or competitor.description
            competitor.why_dangerous = why or competitor.why_dangerous
            hq = _as_str(item.get("headquarters_country") or item.get("headquarters"))
            if hq:
                competitor.headquarters = hq
            if not competitor.is_pinned:
                competitor.threat_level = threat if threat in {"medium", "high"} else "high"
                competitor.overlap_score = max(overlap, competitor.overlap_score or 0)
            competitor.is_tracking = True
        else:
            competitor = Competitor(
                agency_id=agency.id,
                client_id=client.id,
                name=name,
                website=website,
                description=why or None,
                why_dangerous=why or None,
                headquarters=_as_str(item.get("headquarters_country") or item.get("headquarters")) or None,
                threat_level=threat if threat in {"medium", "high"} else "high",
                overlap_score=overlap,
                is_tracking=True,
            )
            db.add(competitor)
            created_competitors += 1

    await db.flush()
    return {
        "features": len(feature_rows),
        "competitors_added": created_competitors,
        "competitors_requested": count,
        "competitor_scope": scope,
        "competitor_country": country or None,
        "competitors_kept_existing": len(protected_existing),
        "competitors_pruned_global": pruned_global,
        "competitor_mode": mode,
        "goals": len(client.goals or []),
        "industry": client.industry,
        "niche": client.niche,
        "market_area": market_area,
        "business_model": business_model,
    }



async def run_competitive_pack(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    competitor_scope: str = "local",
    competitor_country: str | None = None,
    competitor_count: int = 5,
    competitor_mode: str = "add",
) -> dict:
    count = max(1, min(10, int(competitor_count or 5)))
    raw_mode = str(competitor_mode or "add").strip().lower()
    mode = raw_mode if raw_mode in {"update", "add", "replace"} else "add"
    scope = "global" if str(competitor_scope).lower() == "global" else "local"
    required_market = _as_str(competitor_country).strip() or _market_area_from_client(client)
    features = (
        await db.execute(
            select(ProductFeature).where(
                ProductFeature.client_id == client.id,
                ProductFeature.agency_id == agency.id,
            )
        )
    ).scalars().all()
    competitors = (
        await db.execute(
            select(Competitor).where(
                Competitor.client_id == client.id,
                Competitor.agency_id == agency.id,
                Competitor.is_tracking.is_(True),
            )
        )
    ).scalars().all()

    # Standalone pack calls may still need an enrich when the client has no rivals yet.
    if not features or not competitors:
        await enrich_client_profile(
            db,
            agency,
            client,
            competitor_scope=competitor_scope,
            competitor_country=competitor_country,
            competitor_count=count,
            competitor_mode=mode if competitors else ("add" if mode == "update" else mode),
        )
        features = (
            await db.execute(
                select(ProductFeature).where(
                    ProductFeature.client_id == client.id,
                    ProductFeature.agency_id == agency.id,
                )
            )
        ).scalars().all()
        competitors = (
            await db.execute(
                select(Competitor).where(
                    Competitor.client_id == client.id,
                    Competitor.agency_id == agency.id,
                    Competitor.is_tracking.is_(True),
                )
            )
        ).scalars().all()

    # update: refresh up to `count` existing. add: keep all tracked. replace: pinned + up to `count` fresh.
    pinned = [c for c in competitors if c.is_pinned]
    others = sorted(
        [c for c in competitors if not c.is_pinned],
        key=lambda c: float(c.overlap_score or 0),
        reverse=True,
    )
    if mode == "update":
        competitors = (pinned + others)[: max(count, len(pinned))]
    elif mode == "replace":
        competitors = pinned + others[:count]
    else:
        competitors = pinned + others

    if not features or not competitors:
        if not features and not competitors:
            raise ValueError(
                "Could not build features or rivals for this client. Add a website, then run intel again "
                "or add features and competitors manually."
            )
        if not features:
            raise ValueError(
                "Could not extract product features for this client. Add a website, then run intel again "
                "or add features manually."
            )
        raise ValueError(
            "No matching competitors survived quality filters. "
            "Check SerpAPI key under Integrations/BYOK (search is returning unauthorized), "
            "or add software-house rivals manually and pin them."
        )

    kept: list[Competitor] = []
    analyzed: list[Competitor] = []
    for competitor in competitors:
        # Normalize stored website so UI links open correctly
        if competitor.website:
            competitor.website = _normalize_website(competitor.website) or competitor.website
        site_data = {}
        if competitor.website:
            site_data = await scrape_website(db, agency.id, competitor.website)
        site_md = (site_data.get("markdown") or "")[:3500]
        analysis = await ai_service.structured_json(
            db,
            agency.id,
            (
                "Enrich a competitor for competitive intelligence against THIS client only. "
                "Return JSON keys: tagline, description, headquarters, headquarters_country, industry, business_model, "
                "overlap_score (0-100), threat_level (low|medium|high), is_leading_rival (boolean), "
                "same_niche (boolean), same_market (boolean), is_global_platform (boolean), "
                "why_dangerous (1-2 sentences), evidence_snippet (short quote/paraphrase from site), "
                "features (array of {name, category, description}). "
                "Score overlap high ONLY when industry + niche + buyer + business model truly match. "
                "If the rival is a consumer shopping/ecommerce retailer, fintech wallet/payments app, bank, "
                "government board/ministry/authority, news site, directory, or unrelated industry, "
                "set same_niche=false, is_leading_rival=false, overlap_score below 40. "
                "Same broad industry label like 'Technology' is NOT enough — buyers and product must match. "
                "If the site is a directory, review site, news article, job board, or unrelated industry, "
                "set same_niche=false, is_leading_rival=false, overlap_score below 40. "
                "Global hyperscalers/platforms that are not peer businesses should be low threat, "
                "same_niche=false, is_global_platform=true. "
                + (
                    f"LOCAL MARKET REQUIRED: {required_market}. "
                    f"Set headquarters_country from SITE EVIDENCE only (not guesses). "
                    f"If headquarters / primary selling country is clearly NOT {required_market}, "
                    "set same_market=false and overlap_score below 40. "
                    "Do not treat neighboring countries (e.g. India vs Pakistan, Singapore vs Pakistan) as the same market. "
                    "Never invent that a foreign company sells primarily in the required market."
                    if scope == "local" and required_market
                    else ""
                )
            ),
            json.dumps(
                {
                    "client": client.name,
                    "client_industry": client.industry,
                    "client_niche": client.niche,
                    "client_business_model": _business_model_from_client(client),
                    "client_market_area": required_market or _market_area_from_client(client),
                    "competitor_scope": scope,
                    "required_market": required_market or None,
                    "client_features": [
                        {"name": f.name, "category": f.category, "description": f.description} for f in features
                    ],
                    "competitor": {
                        "name": competitor.name,
                        "website": competitor.website,
                        "site_excerpt": site_md,
                    },
                }
            )[:9000],
            temperature=0.2,
        )
        # If AI fallback text returned, keep prior competitor values — but do not auto-trust as leading
        if "summary" in analysis and "features" not in analysis:
            analysis = {
                "overlap_score": competitor.overlap_score or 55,
                "threat_level": competitor.threat_level or "medium",
                "is_leading_rival": False,
                "same_niche": True if competitor.is_pinned else None,
                "why_dangerous": competitor.why_dangerous or competitor.description or f"{competitor.name} competes for the same buyers.",
                "features": competitor.feature_list or [],
            }

        competitor.tagline = _as_str(analysis.get("tagline")) or competitor.tagline
        competitor.description = _as_str(analysis.get("description")) or competitor.description
        competitor.headquarters = _as_str(analysis.get("headquarters") or analysis.get("headquarters_country")) or competitor.headquarters
        try:
            competitor.overlap_score = float(analysis.get("overlap_score") or competitor.overlap_score or 55)
        except (TypeError, ValueError):
            competitor.overlap_score = competitor.overlap_score or 55
        competitor.threat_level = _as_str(analysis.get("threat_level") or competitor.threat_level or "medium").lower()
        if competitor.threat_level not in {"low", "medium", "high"}:
            competitor.threat_level = "medium"
        competitor.feature_list = analysis.get("features") if isinstance(analysis.get("features"), list) else (competitor.feature_list or [])
        competitor.why_dangerous = _as_str(analysis.get("why_dangerous")) or competitor.why_dangerous
        competitor.evidence_snippet = _as_str(analysis.get("evidence_snippet")) or competitor.evidence_snippet
        if site_md and not competitor.evidence_snippet:
            competitor.evidence_snippet = site_md[:280]
        competitor.last_scraped_at = datetime.utcnow()
        analyzed.append(competitor)

        # Trust site + HQ fields for geo — AI blurbs often hallucinate the client's country
        site_geo_blob = " ".join(
            [
                _as_str(competitor.headquarters),
                _as_str(analysis.get("headquarters_country")),
                site_md[:2000],
            ]
        ).lower()
        hq_key = _normalize_country_key(_as_str(analysis.get("headquarters_country") or competitor.headquarters))
        # If site text clearly names another country, prefer that over AI HQ claim
        site_conflict = _mentions_conflicting_country(site_md[:2000].lower(), competitor.website, required_market) if required_market else False
        if site_conflict:
            for key, aliases in _COUNTRY_ALIASES.items():
                if key == _normalize_country_key(required_market):
                    continue
                if _blob_mentions_any(site_md[:2000].lower(), aliases) or _host_matches_tlds(
                    _domain_of(competitor.website or ""), _COUNTRY_TLDS.get(key, set())
                ):
                    hq_key = key
                    break
        market_key = _normalize_country_key(required_market)
        peer_blob = " ".join(
            [
                _as_str(competitor.name),
                _as_str(competitor.description),
                _as_str(analysis.get("industry")),
                _as_str(analysis.get("business_model")),
                site_md[:2000],
            ]
        ).lower()
        bad_peer = _incompatible_peer(
            client_model=_business_model_from_client(client),
            client_industry=_as_str(client.industry),
            client_niche=_as_str(client.niche),
            rival_model=_as_str(analysis.get("business_model")),
            rival_industry=_as_str(analysis.get("industry")),
            rival_blob=peer_blob,
        )
        curated = _is_curated_seed_rival(competitor.name, required_market)
        # Curated seeds already passed geo/niche gates — don't let flaky scrape/AI wipe the list down to 1
        if curated and not hq_key and market_key:
            hq_key = market_key
            if not competitor.headquarters:
                competitor.headquarters = required_market
        has_local_proof = (
            curated
            or (bool(hq_key) and bool(market_key) and hq_key == market_key)
            or _mentions_target_market(site_geo_blob, competitor.website, required_market)
        )
        wrong_market = (
            scope == "local"
            and bool(required_market)
            and not competitor.is_pinned
            and not curated
            and (
                analysis.get("same_market") is False
                or site_conflict
                or _mentions_conflicting_country(site_geo_blob, competitor.website, required_market)
                or (bool(hq_key) and bool(market_key) and hq_key != market_key)
                or not has_local_proof
            )
        )
        # Soften dead-site drop when Firecrawl fails but HQ/market already look local
        dead_site = (
            not competitor.is_pinned
            and not curated
            and (
                not competitor.website
                or (
                    bool(competitor.website)
                    and _site_looks_parked_or_empty(site_md)
                    and site_data.get("status") != "ok"
                )
                or (
                    bool(competitor.website)
                    and site_data.get("status") == "error"
                    and not site_md.strip()
                    and not has_local_proof
                )
            )
        )
        client_is_software_peer = any(
            tok in f"{_as_str(client.industry)} {_as_str(client.niche)} {_business_model_from_client(client)}".lower()
            for tok in ("software", "agency", "it services", "development", "digital", "saas")
        )
        weak_software_peer = (
            client_is_software_peer
            and not competitor.is_pinned
            and not curated
            and bool(site_md)
            and not _site_supports_software_peer(site_md)
        )
        fake_brand = (not competitor.is_pinned) and _is_generic_or_fake_rival_name(competitor.name)
        site_host_noise = bool(competitor.website and _is_serp_noise_domain(competitor.website))
        off_niche = (
            not competitor.is_pinned
            and (
                fake_brand
                or _is_global_megarival(competitor.name, competitor.website)
                or site_host_noise
                or (analysis.get("is_global_platform") is True and not curated)
                or (analysis.get("same_niche") is False and not curated)
                or (bad_peer and not curated)
                or wrong_market
                or dead_site
                or weak_software_peer
                or ((competitor.overlap_score or 0) < 55 and not curated)
                or (
                    not curated
                    and competitor.threat_level == "low"
                    and (competitor.overlap_score or 0) < 65
                    and analysis.get("is_leading_rival") is False
                )
            )
        )
        if off_niche:
            competitor.is_tracking = False
            competitor.threat_level = "low"
            continue

        competitor.is_tracking = True
        if competitor.threat_level == "low":
            competitor.threat_level = "medium"
        kept.append(competitor)

    # Always put pinned/manual rivals back even if AI scored them weakly
    kept_ids = {c.id for c in kept}
    for competitor in analyzed:
        if competitor.is_pinned and competitor.id not in kept_ids:
            competitor.is_tracking = True
            if competitor.threat_level == "low":
                competitor.threat_level = "medium"
            if (competitor.overlap_score or 0) < 55:
                competitor.overlap_score = 70.0
            kept.insert(0, competitor)
            kept_ids.add(competitor.id)

    if not kept and analyzed:
        # Prefer strongest overlaps that are not megacorp/noise domains.
        # For local runs, never resurrect clear foreign-market rivals as a fallback.
        def _fallback_ok(c: Competitor) -> bool:
            if _is_generic_or_fake_rival_name(c.name):
                return False
            if _is_global_megarival(c.name, c.website):
                return False
            if c.website and _is_serp_noise_domain(c.website):
                return False
            if scope == "local" and required_market:
                blob = f"{c.headquarters or ''} {c.description or ''} {c.why_dangerous or ''}".lower()
                if _mentions_conflicting_country(blob, c.website, required_market):
                    return False
            return True

        analyzed_sorted = sorted(
            [c for c in analyzed if _fallback_ok(c)] or [],
            key=lambda c: float(c.overlap_score or 0),
            reverse=True,
        )
        for competitor in analyzed_sorted[:count]:
            competitor.is_tracking = True
            if competitor.threat_level not in {"medium", "high"}:
                competitor.threat_level = "medium"
            kept.append(competitor)

    # update: refresh up to `count`. replace: pinned + up to `count` fresh. add: keep all after prune.
    kept = sorted(kept, key=lambda c: (1 if c.is_pinned else 0, float(c.overlap_score or 0)), reverse=True)
    pinned_final = [c for c in kept if c.is_pinned]
    others_final = [c for c in kept if not c.is_pinned]
    if mode == "update":
        competitors = pinned_final + others_final[: max(0, count - len(pinned_final))]
    elif mode == "replace":
        # Untrack extras that survived enrich but exceed the fresh-set size
        for extra in others_final[count:]:
            if not extra.is_pinned:
                extra.is_tracking = False
        competitors = pinned_final + others_final[:count]
    else:
        competitors = pinned_final + others_final
    if not competitors:
        raise ValueError("No competitors available for this client after enrichment.")

    comparisons_payload: list[dict] = []
    for competitor in competitors:
        block = await _generate_competitor_comparisons(db, agency, client, list(features), competitor)
        comparisons_payload.append(block)

    pack = await ai_service.structured_json(
        db,
        agency.id,
        (
            "Build gap reports and goal-weighted alerts. "
            "Return JSON with keys: "
            "gap_reports (array of {competitor_name, summary, leading[], lagging[], opportunities[]}), "
            "goal_alerts (array of {goal, title, why_it_matters, impact, action, content_draft, estimated_cost, competitor_trigger, missing_feature}), "
            "highlights (string array of sharp executive takeaways). "
            "impact MUST be exactly one of: low | medium | high (never a sentence). "
            "ALERT RULE: only create alerts for features/specialties competitors have that the client does NOT have. "
            "Do not alert on features the client already owns. Be specific. No generic filler."
        ),
        json.dumps(
            {
                "client": {
                    "name": client.name,
                    "industry": client.industry,
                    "niche": client.niche,
                    "tagline": client.tagline,
                    "goals": client.goals or [],
                    "features": [
                        {"name": f.name, "category": f.category, "description": f.description} for f in features
                    ],
                },
                "competitors": [
                    {
                        "name": c.name,
                        "overlap_score": c.overlap_score,
                        "threat_level": c.threat_level,
                        "tagline": c.tagline,
                        "features": c.feature_list or [],
                    }
                    for c in competitors
                ],
                "comparison_snapshot": comparisons_payload,
            }
        )[:14000],
        temperature=0.35,
    )
    pack["comparisons"] = comparisons_payload

    await db.execute(delete(FeatureComparison).where(FeatureComparison.client_id == client.id))
    await db.execute(delete(GapReport).where(GapReport.client_id == client.id))
    await db.execute(delete(GoalAlert).where(GoalAlert.client_id == client.id))

    name_to_comp = {_as_str(c.name).lower(): c for c in competitors}
    comparison_count = 0
    for block in pack.get("comparisons", []):
        comp = name_to_comp.get(_as_str(block.get("competitor_name")).lower())
        if not comp:
            continue
        for row in block.get("rows", [])[:10]:
            cleaned = _normalize_comparison_row(row, client.name, comp.name)
            if not cleaned:
                continue
            db.add(
                FeatureComparison(
                    agency_id=agency.id,
                    client_id=client.id,
                    competitor_id=comp.id,
                    competitor_name=comp.name,
                    feature_name=cleaned["feature_name"],
                    category=cleaned["category"],
                    our_status=cleaned["our_status"],
                    competitor_status=cleaned["competitor_status"],
                    note=cleaned["note"],
                    how_competitor_leads=cleaned["how_competitor_leads"],
                    how_to_improve=cleaned["how_to_improve"],
                    citations=cleaned["citations"],
                    confidence_score=cleaned["confidence_score"],
                    evidence_strength=cleaned["evidence_strength"],
                    is_contested_move=cleaned["is_contested_move"],
                )
            )
            comparison_count += 1

    gap_count = 0
    for gap in pack.get("gap_reports", []) or []:
        comp = name_to_comp.get(_as_str(gap.get("competitor_name")).lower())
        if not comp:
            continue
        summary = _as_str(gap.get("summary")).strip()
        if not summary:
            continue
        leading = gap.get("leading") if isinstance(gap.get("leading"), list) else []
        lagging = gap.get("lagging") if isinstance(gap.get("lagging"), list) else []
        opportunities = gap.get("opportunities") if isinstance(gap.get("opportunities"), list) else []
        citations = gap.get("citations") if isinstance(gap.get("citations"), list) else []
        if not citations and comp.website:
            citations = [
                {
                    "url": comp.website or "",
                    "snippet": _as_str(comp.evidence_snippet or comp.description)[:300],
                    "source": "website",
                }
            ]
        try:
            conf = float(gap.get("confidence_score") or 0.6)
        except (TypeError, ValueError):
            conf = 0.6
        db.add(
            GapReport(
                agency_id=agency.id,
                client_id=client.id,
                competitor_id=comp.id,
                competitor_name=comp.name,
                summary=summary,
                leading=leading,
                lagging=lagging,
                opportunities=opportunities,
                citations=citations,
                confidence_score=conf,
                evidence_strength=_as_str(gap.get("evidence_strength"), "medium"),
            )
        )
        gap_count += 1

    if gap_count == 0:
        for comp in competitors:
            rival_rows = [b for b in comparisons_payload if _as_str(b.get("competitor_name")).lower() == comp.name.lower()]
            leading_feats = []
            opportunities = []
            for block in rival_rows:
                for row in block.get("rows") or []:
                    cleaned = row if "our_status" in row and "feature_name" in row else None
                    if not cleaned:
                        continue
                    if cleaned.get("competitor_status") == "leading":
                        leading_feats.append(cleaned["feature_name"])
                    if cleaned.get("our_status") == "lagging":
                        opportunities.append(cleaned.get("how_to_improve") or f"Improve {cleaned['feature_name']}")
            if not leading_feats and comp.feature_list:
                for f in comp.feature_list[:5]:
                    if isinstance(f, dict) and f.get("name"):
                        leading_feats.append(_as_str(f.get("name")))
            summary = (
                f"{comp.name} leads on {', '.join(leading_feats[:4])}."
                if leading_feats
                else f"{comp.name} remains a high-overlap rival ({int(comp.overlap_score or 0)}% overlap) that can pressure {client.name} in deals."
            )
            db.add(
                GapReport(
                    agency_id=agency.id,
                    client_id=client.id,
                    competitor_id=comp.id,
                    competitor_name=comp.name,
                    summary=summary,
                    leading=leading_feats[:8],
                    lagging=[],
                    opportunities=(opportunities or [f"Build a sharper counter-narrative vs {comp.name}"])[:8],
                    citations=[
                        {
                            "url": comp.website or "",
                            "snippet": _as_str(comp.evidence_snippet or comp.why_dangerous or comp.description)[:300],
                            "source": "website",
                        }
                    ]
                    if comp.website
                    else [],
                    confidence_score=0.62,
                    evidence_strength="medium",
                )
            )
            gap_count += 1

    alert_count = 0
    for alert in pack.get("goal_alerts", []) or []:
        title = _as_str(alert.get("title"), "Goal alert").strip()
        why = _as_str(alert.get("why_it_matters")).strip()
        action = _as_str(alert.get("action")).strip()
        if not title or not why:
            continue
        citations = alert.get("citations") if isinstance(alert.get("citations"), list) else []
        try:
            conf = float(alert.get("confidence_score") or 0.6)
        except (TypeError, ValueError):
            conf = 0.6
        db.add(
            GoalAlert(
                agency_id=agency.id,
                client_id=client.id,
                goal=_clip(_as_str(alert.get("goal") or ((client.goals or ["Grow market share"])[0])), 500),
                title=_clip(title, 500),
                why_it_matters=why,
                impact=_level_label(alert.get("impact"), "medium", max_len=255),
                action=action or f"Prioritize a response to {title}",
                content_draft=_as_str(alert.get("content_draft")),
                estimated_cost=_clip(_as_str(alert.get("estimated_cost")), 120),
                competitor_trigger=_clip(
                    _as_str(alert.get("competitor_trigger") or alert.get("missing_feature")), 255
                ),
                citations=citations,
                confidence_score=conf,
                evidence_strength=_level_label(alert.get("evidence_strength"), "medium"),
            )
        )
        alert_count += 1

    if alert_count == 0:
        seen_alert: set[str] = set()
        client_feat_names = {f.name.lower() for f in features}

        def _add_specialty_alert(
            *,
            feat: str,
            comp_name: str,
            why: str,
            action: str,
            citations: list | None = None,
            confidence: float = 0.6,
            evidence: str = "medium",
        ) -> None:
            nonlocal alert_count
            key = feat.lower()
            if not feat or key in seen_alert or alert_count >= 8:
                return
            if key in client_feat_names:
                return
            seen_alert.add(key)
            db.add(
                GoalAlert(
                    agency_id=agency.id,
                    client_id=client.id,
                    goal=_clip(((client.goals or ["Close competitive gaps"])[0]), 500),
                    title=_clip(f"Missing specialty: {feat}", 500),
                    why_it_matters=why,
                    impact="high",
                    action=action,
                    content_draft=f"Buyers comparing you to {comp_name} will ask about {feat}. Prepare a gap-close narrative this week.",
                    estimated_cost="1-2 sprints",
                    competitor_trigger=_clip(comp_name, 255),
                    citations=citations or [],
                    confidence_score=confidence,
                    evidence_strength=_level_label(evidence, "medium"),
                )
            )
            alert_count += 1

        for block in comparisons_payload:
            comp_name = _as_str(block.get("competitor_name"))
            for row in block.get("rows") or []:
                feat = _as_str(row.get("feature_name")).strip()
                our = _as_str(row.get("our_status")).lower()
                theirs = _as_str(row.get("competitor_status")).lower()
                if theirs != "leading" and our not in {"lagging", "missing", "weak", "none", "absent"}:
                    continue
                try:
                    conf = float(row.get("confidence_score") or 0.6)
                except (TypeError, ValueError):
                    conf = 0.6
                _add_specialty_alert(
                    feat=feat,
                    comp_name=comp_name,
                    why=_as_str(row.get("how_competitor_leads"))
                    or f"{comp_name} has {feat} as a specialty you lack or lag on.",
                    action=_as_str(row.get("how_to_improve")) or f"Add {feat} to wishlist and ship a development plan.",
                    citations=row.get("citations") if isinstance(row.get("citations"), list) else [],
                    confidence=conf,
                    evidence=_as_str(row.get("evidence_strength"), "medium"),
                )
                if alert_count >= 8:
                    break
            if alert_count >= 8:
                break

        if alert_count == 0:
            for comp in competitors:
                for f in comp.feature_list or []:
                    name = _as_str(f.get("name") if isinstance(f, dict) else f).strip()
                    _add_specialty_alert(
                        feat=name,
                        comp_name=comp.name,
                        why=f"{comp.name} lists {name} as a product specialty that {client.name} does not currently advertise.",
                        action=f"Add {name} to wishlist and draft a development plan this week.",
                        citations=[
                            {
                                "url": comp.website or "",
                                "snippet": _as_str(comp.evidence_snippet or comp.description or name)[:300],
                                "source": "website",
                            }
                        ]
                        if comp.website
                        else [],
                    )
                    if alert_count >= 8:
                        break
                if alert_count >= 8:
                    break

    await db.flush()
    return {
        "competitors": len(competitors),
        "comparisons": comparison_count,
        "gaps": gap_count,
        "alerts": alert_count,
        "highlights": pack.get("highlights") or [],
    }


async def love_feature_and_build_tickets(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    feature: ProductFeature,
) -> list[FeatureTicket]:
    feature.is_loved = True
    feature.is_wishlisted = True
    comparisons = (
        await db.execute(
            select(FeatureComparison).where(
                FeatureComparison.client_id == client.id,
                FeatureComparison.agency_id == agency.id,
                FeatureComparison.feature_name == feature.name,
            )
        )
    ).scalars().all()
    gaps = (
        await db.execute(
            select(GapReport).where(GapReport.client_id == client.id, GapReport.agency_id == agency.id)
        )
    ).scalars().all()
    alerts = (
        await db.execute(
            select(GoalAlert).where(GoalAlert.client_id == client.id, GoalAlert.agency_id == agency.id)
        )
    ).scalars().all()

    evidence = []
    for c in comparisons:
        for cite in c.citations or []:
            evidence.append(cite)
        evidence.append(
            {
                "url": "",
                "snippet": f"{c.competitor_name}: {c.how_competitor_leads}"[:350],
                "source": "comparison",
            }
        )

    # Keep AI under platform proxy limits (~60s). Fall back to templates on timeout/failure.
    payload: dict = {}
    try:
        payload = await asyncio.wait_for(
            ai_service.structured_json(
                db,
                agency.id,
                (
                    "The marketing agency / client manually selected this feature. Create BOARD-READY Jira work. "
                    "Return JSON {tickets:[{heading, body, acceptance_criteria[], priority, ticket_type, labels[], "
                    "estimated_effort, story_points, why_useful, competitor_context, evidence_links:[{url, snippet, source}]}]}. "
                    "Requirements:\n"
                    "- Exactly 1 epic first, then 4-6 stories/tasks under that theme.\n"
                    "- ticket_type must be epic|story|task.\n"
                    "- Each story must have 3-5 measurable acceptance criteria.\n"
                    "- Include effort estimate and story_points (1-8).\n"
                    "- Link competitor evidence in competitor_context and evidence_links.\n"
                    "- Cover: packaging, GTM, sales enablement, demo, analytics.\n"
                    "- No filler. Valid, shippable tickets only. Keep responses concise."
                ),
                json.dumps(
                    {
                        "feature": {
                            "name": feature.name,
                            "category": feature.category,
                            "description": feature.description,
                        },
                        "client": {"name": client.name, "goals": (client.goals or [])[:5]},
                        "feature_comparisons": [
                            {
                                "competitor": c.competitor_name,
                                "our_status": c.our_status,
                                "competitor_status": c.competitor_status,
                                "how_to_improve": (c.how_to_improve or "")[:280],
                                "how_competitor_leads": (c.how_competitor_leads or "")[:280],
                                "confidence_score": c.confidence_score,
                            }
                            for c in comparisons[:6]
                        ],
                        "related_gaps": [
                            {
                                "competitor": g.competitor_name,
                                "opportunities": (g.opportunities or [])[:3],
                                "summary": (g.summary or "")[:220],
                            }
                            for g in gaps[:4]
                        ],
                        "related_alerts": [
                            {"title": a.title, "action": (a.action or "")[:160]} for a in alerts[:4]
                        ],
                        "seed_evidence": evidence[:6],
                    }
                )[:7000],
                temperature=0.3,
            ),
            timeout=35,
        )
    except asyncio.TimeoutError:
        logger.warning("development-plan AI timed out for feature=%s — using templates", feature.id)
        payload = {}
    except Exception:
        logger.exception("development-plan AI failed for feature=%s — using templates", feature.id)
        payload = {}

    await db.execute(delete(FeatureTicket).where(FeatureTicket.feature_id == feature.id))
    tickets: list[FeatureTicket] = []
    epic_id: str | None = None
    items = [i for i in _as_list(payload.get("tickets")) if isinstance(i, dict)]
    if not any(_as_str(i.get("ticket_type")).lower() == "epic" for i in items):
        items = [
            {
                "heading": f"[Epic] Ship {feature.name} competitive response",
                "body": f"Coordinate product, marketing, and sales work to close gaps around {feature.name}.",
                "acceptance_criteria": [
                    "All child stories completed or explicitly deferred",
                    "Weekly brief includes progress vs named rivals",
                    "Agency can demo the packaged narrative",
                ],
                "priority": "high",
                "ticket_type": "epic",
                "labels": [feature.category, "loved-feature", "epic"],
                "estimated_effort": "2-3 sprints",
                "story_points": 0,
                "why_useful": "Creates one parent workstream for the loved feature.",
                "competitor_context": "Derived from contested competitor comparisons.",
                "evidence_links": evidence[:4],
            }
        ] + items

    story_count = sum(1 for i in items if _as_str(i.get("ticket_type")).lower() != "epic")
    if story_count < 5:
        rival_names = sorted({c.competitor_name for c in comparisons}) or ["top rival"]
        templates = [
            ("story", f"Package {feature.name} as a sellable offer", "Rewrite offer page and sales one-pager with proof points.", ["Offer page live", "One-pager approved", "Proof points cited"], "3-5 days", 5),
            ("story", f"Build competitive battlecard vs {rival_names[0]}", f"Document how {feature.name} beats or matches {rival_names[0]}.", ["Battlecard in shared drive", "Sales team briefed", "Objection responses included"], "2-3 days", 3),
            ("story", f"Ship demo narrative for {feature.name}", "Create a 5-minute demo script with talk track and screens.", ["Script reviewed", "Demo recorded", "AE can run unassisted"], "3-4 days", 5),
            ("story", f"Create GTM messaging kit for {feature.name}", "Homepage module, email, LinkedIn, and paid ad variants.", ["4 assets drafted", "Brand review done", "UTM naming set"], "4-5 days", 5),
            ("story", f"Close product gap called out in contested moves", "Implement the highest-confidence gap tied to this feature.", ["Gap ticket scoped", "Acceptance tests pass", "Changelog published"], "1-2 weeks", 8),
            ("task", f"Collect evidence screenshots for {feature.name}", "Capture rival pages and client proof for citations.", ["At least 5 screenshots", "URLs logged", "Shared with report"], "1 day", 2),
        ]
        existing_heads = {_as_str(i.get("heading")).lower() for i in items}
        for ttype, heading, body, criteria, effort, points in templates:
            if heading.lower() in existing_heads:
                continue
            items.append(
                {
                    "heading": heading,
                    "body": body,
                    "acceptance_criteria": criteria,
                    "priority": "high" if ttype == "story" else "medium",
                    "ticket_type": ttype,
                    "labels": [feature.category, "loved-feature", ttype],
                    "estimated_effort": effort,
                    "story_points": points,
                    "why_useful": f"Board-ready work to commercialize {feature.name} against high-risk rivals.",
                    "competitor_context": f"Rivals in scope: {', '.join(rival_names[:4])}",
                    "evidence_links": evidence[:4],
                }
            )
            existing_heads.add(heading.lower())
            if sum(1 for i in items if _as_str(i.get("ticket_type")).lower() != "epic") >= 6:
                break

    for item in items[:8]:
        ttype = _as_str(item.get("ticket_type"), "story").lower()
        if ttype not in {"epic", "story", "task"}:
            ttype = "story"
        criteria = [_as_str(c) for c in _as_list(item.get("acceptance_criteria"))]
        if ttype != "epic" and len(criteria) < 3:
            criteria = criteria + [
                "Definition of done reviewed with agency lead",
                "Competitor evidence linked",
                "Deliverable shared with client stakeholder",
            ]
            criteria = criteria[:6]
        ticket = FeatureTicket(
            agency_id=agency.id,
            client_id=client.id,
            feature_id=feature.id,
            heading=_clip(_as_str(item.get("heading")), 500) or f"Improve {feature.name}"[:500],
            body=_as_str(item.get("body")),
            acceptance_criteria=criteria,
            priority=_level_label(item.get("priority"), "medium", max_len=20),
            ticket_type=ttype,
            labels=[_as_str(l) for l in _as_list(item.get("labels"))] or [feature.category, "loved-feature"],
            estimated_effort=_clip(_as_str(item.get("estimated_effort")), 80),
            story_points=_as_int(item.get("story_points")),
            why_useful=_as_str(item.get("why_useful")),
            competitor_context=_as_str(item.get("competitor_context")),
            evidence_links=_as_list(item.get("evidence_links")) or evidence[:4],
            parent_ticket_id=None if ttype == "epic" else epic_id,
            status="draft",
        )
        db.add(ticket)
        await db.flush()
        if ttype == "epic" and epic_id is None:
            epic_id = ticket.id
        tickets.append(ticket)
    await db.flush()
    return tickets


async def create_all_feature_tickets_in_jira(
    db: AsyncSession,
    agency_id: str,
    client_id: str,
    feature_id: str,
) -> list[FeatureTicket]:
    connected = (
        await db.execute(
            select(Integration).where(
                Integration.agency_id == agency_id,
                Integration.provider == "jira",
                Integration.is_connected.is_(True),
            )
        )
    ).scalar_one_or_none()
    if not connected or not connected.encrypted_credentials:
        raise ValueError("Connect your Jira account first under Integrations.")

    tickets = (
        await db.execute(
            select(FeatureTicket)
            .where(
                FeatureTicket.feature_id == feature_id,
                FeatureTicket.client_id == client_id,
                FeatureTicket.agency_id == agency_id,
            )
            .order_by(FeatureTicket.created_at.asc())
        )
    ).scalars().all()
    if not tickets:
        raise ValueError("No feature tickets found. Generate a development plan first.")

    epic_jira_key = None
    for ticket in tickets:
        if ticket.jira_key and ticket.ticket_type == "epic":
            epic_jira_key = ticket.jira_key
            break

    errors: list[str] = []
    for ticket in tickets:
        if ticket.jira_key:
            if ticket.ticket_type == "epic" and not epic_jira_key:
                epic_jira_key = ticket.jira_key
            continue
        criteria = "\n".join(f"- {c}" for c in (ticket.acceptance_criteria or []))
        evidence = "\n".join(
            f"- {e.get('source', 'source')}: {e.get('url', '')} :: {(e.get('snippet') or '')[:180]}"
            for e in (ticket.evidence_links or [])
            if isinstance(e, dict)
        )
        description = (
            f"{ticket.body}\n\n"
            f"Why useful:\n{ticket.why_useful}\n\n"
            f"Competitor evidence:\n{ticket.competitor_context}\n\n"
            f"Evidence links:\n{evidence or '- n/a'}\n\n"
            f"Acceptance criteria:\n{criteria}\n\n"
            f"Type: {ticket.ticket_type} | Priority: {ticket.priority} | "
            f"Effort: {ticket.estimated_effort} | Points: {ticket.story_points}\n"
            f"Labels: {', '.join(ticket.labels or [])}"
        )
        try:
            created = await jira_service.create_jira_ticket(
                db,
                agency_id,
                client_id,
                ticket.heading,
                description,
                insight_id=ticket.id,
                issue_type="Epic" if ticket.ticket_type == "epic" else ("Story" if ticket.ticket_type == "story" else "Task"),
                parent_epic_key=None if ticket.ticket_type == "epic" else epic_jira_key,
            )
            ticket.jira_key = created.jira_key
            ticket.jira_url = created.jira_url
            ticket.jira_epic_key = epic_jira_key
            ticket.status = "created"
            if ticket.ticket_type == "epic":
                epic_jira_key = created.jira_key
                ticket.jira_epic_key = created.jira_key
            await db.flush()
        except Exception as exc:
            logger.warning("Jira push failed for ticket=%s: %s", ticket.id, exc)
            errors.append(f"{ticket.heading[:60]}: {exc}")
            # Don't abort the whole batch — continue with remaining tickets
            continue

    await db.flush()
    pushed = sum(1 for t in tickets if t.jira_key)
    if pushed == 0 and errors:
        raise ValueError(errors[0] if len(errors) == 1 else f"Jira push failed ({len(errors)} errors). First: {errors[0]}")
    return list(tickets)


async def run_full_ai_pipeline(
    db: AsyncSession,
    agency: Agency,
    client: ClientBrand,
    *,
    push_jira: bool = True,
    generate_report: bool = True,
    competitor_scope: str = "local",
    competitor_country: str | None = None,
    competitor_count: int = 5,
    competitor_mode: str = "add",
) -> dict:
    job = TrackingJob(
        agency_id=agency.id,
        client_id=client.id,
        job_type="full_ai_pipeline",
        status=JobStatus.running,
        started_at=datetime.utcnow(),
        detail="Autonomous AI pipeline running",
    )
    db.add(job)
    await db.flush()

    try:
        from app.services.embeddings import index_client_intel
        from app.services.intelligence import run_client_intelligence

        enrich = await enrich_client_profile(
            db,
            agency,
            client,
            competitor_scope=competitor_scope,
            competitor_country=competitor_country,
            competitor_count=competitor_count,
            competitor_mode=competitor_mode,
        )
        pack = await run_competitive_pack(
            db,
            agency,
            client,
            competitor_scope=competitor_scope,
            competitor_country=competitor_country,
            competitor_count=competitor_count,
            competitor_mode=competitor_mode,
        )
        radar = await run_client_intelligence(
            db, agency, client, competitor_country=competitor_country
        )

        report_id = None
        if generate_report:
            report = await generate_client_report(db, agency, client, period_label="AI Auto Brief")
            report_id = report.id

        jira_pushed = 0
        if push_jira:
            wishlisted = (
                await db.execute(
                    select(ProductFeature).where(
                        ProductFeature.client_id == client.id,
                        ProductFeature.agency_id == agency.id,
                        ProductFeature.is_wishlisted.is_(True),
                    )
                )
            ).scalars().all()
            for feature in wishlisted[:5]:
                try:
                    tickets = await love_feature_and_build_tickets(db, agency, client, feature)
                    pushed = await create_all_feature_tickets_in_jira(
                        db, agency.id, client.id, feature.id
                    )
                    jira_pushed += len(pushed or tickets or [])
                except Exception:
                    continue

        indexed = 0
        try:
            async with db.begin_nested():
                indexed = await index_client_intel(db, agency.id, client)
        except Exception as emb_exc:
            logger.warning("index_client_intel skipped: %s", emb_exc)
            indexed = 0

        result = {
            "enrich": enrich,
            "pack": pack,
            "radar_job_id": getattr(radar, "id", None),
            "report_id": report_id,
            "jira_tickets_pushed": jira_pushed,
            "embeddings_indexed": indexed,
            "note": "Wishlist items can auto-push to Jira when push_jira=True and Jira is connected.",
        }
        job.status = JobStatus.completed
        job.finished_at = datetime.utcnow()
        job.result_meta = result
        job.detail = "Autonomous AI pipeline completed"
        await db.flush()
        return result
    except Exception as exc:
        job.status = JobStatus.failed
        job.finished_at = datetime.utcnow()
        job.detail = str(exc)[:800]
        await db.flush()
        raise
