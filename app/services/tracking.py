from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
import logging

import httpx
from apify_client import ApifyClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Agency, ApiKeyVault, Competitor, CompetitorSnapshot, UsageEvent
from app.security import decrypt_secret
from sqlalchemy import select

settings = get_settings()
logger = logging.getLogger("marketbiqs.tracking")

# Cheap Google Trends actor (~$0.80–$1.20 / 1k results). 5 results ≈ well under $0.01.
RADAR_TRENDS_ACTOR = "vnx0/google-trends-scraper"
RADAR_TRENDS_MAX = 5
RADAR_TRENDS_MAX_CHARGE_USD = Decimal("0.01")


async def _vault_key(db: AsyncSession, agency_id: str, provider: str) -> str | None:
    stmt = select(ApiKeyVault).where(
        ApiKeyVault.agency_id == agency_id,
        ApiKeyVault.provider == provider,
        ApiKeyVault.is_active.is_(True),
    )
    result = await db.execute(stmt)
    vault = result.scalar_one_or_none()
    if vault:
        return decrypt_secret(vault.encrypted_key)
    return None


async def resolve_apify(db: AsyncSession, agency_id: str) -> str:
    return (await _vault_key(db, agency_id, "apify")) or settings.apify_key


async def resolve_serp(db: AsyncSession, agency_id: str) -> str:
    return (await _vault_key(db, agency_id, "serpapi")) or settings.serp_api


async def resolve_firecrawl(db: AsyncSession, agency_id: str) -> str:
    return (await _vault_key(db, agency_id, "firecrawl")) or settings.firecrawl_api_key


async def track_usage(db: AsyncSession, agency_id: str, event_type: str, units: int = 1, meta: dict | None = None) -> None:
    db.add(
        UsageEvent(
            agency_id=agency_id,
            event_type=event_type,
            units=units,
            meta=meta or {},
        )
    )
    agency = await db.get(Agency, agency_id)
    if agency:
        agency.scrape_units_used = (agency.scrape_units_used or 0) + max(1, units)


async def ensure_scrape_quota(db: AsyncSession, agency_id: str) -> None:
    agency = await db.get(Agency, agency_id)
    if agency and agency.scrape_units_used >= agency.scrape_quota:
        raise ValueError("Scrape quota exceeded. Purchase client packs or raise scrape quota.")


async def scrape_website(db: AsyncSession, agency_id: str, url: str) -> dict[str, Any]:
    key = await resolve_firecrawl(db, agency_id)
    if not key or not url:
        return {"url": url, "markdown": "", "status": "skipped", "note": "Missing Firecrawl key or URL"}
    try:
        await ensure_scrape_quota(db, agency_id)
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            if response.status_code >= 400:
                return {"url": url, "status": "error", "detail": response.text[:500]}
            data = response.json()
            await track_usage(db, agency_id, "firecrawl_scrape", 1, {"url": url})
            markdown = ((data.get("data") or {}).get("markdown")) or ""
            return {"url": url, "markdown": markdown[:12000], "status": "ok"}
    except Exception as exc:
        return {"url": url, "markdown": "", "status": "error", "detail": str(exc)[:500]}


async def serp_visibility(
    db: AsyncSession,
    agency_id: str,
    query: str,
    *,
    location: str | None = None,
    gl: str | None = None,
) -> dict[str, Any]:
    key = await resolve_serp(db, agency_id)
    if not key:
        return {"query": query, "status": "skipped", "organic": []}
    try:
        params: dict[str, Any] = {"engine": "google", "q": query, "api_key": key, "num": 10}
        if location:
            params["location"] = location
        if gl:
            params["gl"] = gl
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(
                "https://serpapi.com/search.json",
                params=params,
            )
            if response.status_code >= 400:
                logger.warning(
                    "SerpAPI error status=%s query=%s detail=%s",
                    response.status_code,
                    query[:120],
                    response.text[:200],
                )
                return {
                    "query": query,
                    "status": "unauthorized" if response.status_code in {401, 403} else "error",
                    "detail": response.text[:500],
                    "organic": [],
                }
            data = response.json()
            await track_usage(db, agency_id, "serp_search", 1, {"query": query, "location": location, "gl": gl})
            organic = [
                {
                    "position": i.get("position"),
                    "title": i.get("title"),
                    "link": i.get("link"),
                    "snippet": i.get("snippet"),
                }
                for i in data.get("organic_results", [])[:10]
            ]
            return {"query": query, "status": "ok", "organic": organic}
    except Exception as exc:
        return {"query": query, "status": "error", "detail": str(exc)[:500], "organic": []}


async def run_apify_actor(
    db: AsyncSession,
    agency_id: str,
    actor_id: str,
    run_input: dict[str, Any],
    *,
    max_items: int | None = None,
    max_total_charge_usd: Decimal | None = None,
    wait_seconds: int = 90,
    memory_mbytes: int | None = None,
) -> dict[str, Any]:
    token = await resolve_apify(db, agency_id)
    if not token:
        return {"status": "skipped", "items": [], "note": "Missing Apify token"}
    try:
        client = ApifyClient(token)
        call_kwargs: dict[str, Any] = {
            "run_input": run_input,
            "wait_duration": timedelta(seconds=wait_seconds),
        }
        if max_items is not None:
            call_kwargs["max_items"] = max_items
        if max_total_charge_usd is not None:
            call_kwargs["max_total_charge_usd"] = max_total_charge_usd
        if memory_mbytes is not None:
            call_kwargs["memory_mbytes"] = memory_mbytes

        # Newer apify-client uses wait_duration (not timeout_secs) and returns a Run object
        run = client.actor(actor_id).call(**call_kwargs)
        dataset_id = None
        if run is None:
            return {"status": "error", "items": [], "detail": "Apify run returned no result (timeout or failed start)"}
        if isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
            run_status = run.get("status")
        else:
            dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
            run_status = getattr(run, "status", None)
        items: list[dict[str, Any]] = []
        if dataset_id:
            for item in client.dataset(dataset_id).iterate_items(limit=max_items or 25):
                if isinstance(item, dict):
                    items.append(item)
                else:
                    # pydantic/model items → plain dict when possible
                    dumped = getattr(item, "model_dump", None) or getattr(item, "dict", None)
                    items.append(dumped() if callable(dumped) else {"value": str(item)[:500]})
        try:
            await track_usage(db, agency_id, "apify_run", max(1, len(items)), {"actor": actor_id})
        except Exception as usage_exc:
            # Don't fail the scrape if usage accounting cannot flush (e.g. missing agency row)
            logger.warning("Apify usage track failed: %s", usage_exc)
            await db.rollback()
        return {"status": "ok", "items": items, "run_status": str(run_status) if run_status else "SUCCEEDED"}
    except Exception as exc:
        return {"status": "error", "items": [], "detail": str(exc)[:500]}


def _geo_from_text(text: str | None) -> str:
    raw = (text or "").strip().lower()
    mapping = {
        "united states": "US",
        "usa": "US",
        "us": "US",
        "america": "US",
        "united kingdom": "GB",
        "uk": "GB",
        "england": "GB",
        "pakistan": "PK",
        "india": "IN",
        "uae": "AE",
        "united arab emirates": "AE",
        "saudi": "SA",
        "saudi arabia": "SA",
        "canada": "CA",
        "australia": "AU",
        "germany": "DE",
        "singapore": "SG",
        "japan": "JP",
        "korea": "KR",
        "south korea": "KR",
        "france": "FR",
        "brazil": "BR",
    }
    if raw in mapping:
        return mapping[raw]
    for key, code in mapping.items():
        if key in raw:
            return code
    return "US"


async def scrape_radar_top_trends(
    db: AsyncSession,
    agency_id: str,
    *,
    geo: str | None = None,
    country_hint: str | None = None,
    limit: int = RADAR_TRENDS_MAX,
) -> dict[str, Any]:
    """
    Ultra-cheap Radar trends pull via ONE Google Trends Apify actor.

    Cost target: <= ~$0.01 / run (daysBack=1, maxResults=5, charge cap $0.01).
    """
    limit = max(1, min(5, int(limit or 5)))
    geo_code = (geo or _geo_from_text(country_hint) or "US").upper()[:2]
    result = await run_apify_actor(
        db,
        agency_id,
        RADAR_TRENDS_ACTOR,
        {
            "geo": geo_code,
            "daysBack": 1,
            "maxResults": limit,
            # Keep proxy on for Apify cloud reliability; result charge is ~$0.0012 each.
            "useProxy": True,
        },
        max_items=limit,
        max_total_charge_usd=RADAR_TRENDS_MAX_CHARGE_USD,
        wait_seconds=75,
        memory_mbytes=512,
    )
    raw_items = result.get("items") or []
    trends: list[dict[str, Any]] = []
    for item in raw_items[:limit]:
        if not isinstance(item, dict):
            continue
        topic = (
            item.get("title")
            or item.get("query")
            or item.get("displayQuery")
            or item.get("keyword")
            or item.get("term")
            or item.get("trend")
            or ""
        )
        topic = str(topic).strip()
        if not topic:
            continue

        # Prefer actor's 1–1000 popularity score; fall back to traffic labels.
        velocity = 0.0
        if item.get("score") is not None:
            try:
                velocity = float(item.get("score") or 0)
            except (TypeError, ValueError):
                velocity = 0.0
        if velocity <= 0:
            traffic = item.get("trafficValue") or item.get("approxTraffic") or item.get("traffic") or 0
            try:
                velocity = float(str(traffic).replace(",", "").replace("+", "").strip() or 0)
            except (TypeError, ValueError):
                velocity = 50.0
        # Normalize into roughly 0–100 for UI bars
        if velocity > 100:
            velocity = min(100.0, round(velocity / 10.0, 2))

        summary_bits: list[str] = []
        headline = item.get("firstNewsTitle") or item.get("description")
        if headline:
            summary_bits.append(str(headline)[:220])
        news = item.get("newsItems") or item.get("news") or item.get("articles") or []
        if not summary_bits and isinstance(news, list) and news:
            first = news[0] if isinstance(news[0], dict) else {}
            ntitle = first.get("title") or first.get("snippet") or ""
            if ntitle:
                summary_bits.append(str(ntitle)[:220])
        summary = " · ".join(summary_bits) or f"Trending now in {geo_code} (Google Trends)."

        keywords: list[str] = [topic]
        related = item.get("relatedQueries") or []
        if isinstance(related, list):
            for rel in related[:7]:
                if isinstance(rel, str) and rel.strip():
                    keywords.append(rel.strip())
                elif isinstance(rel, dict):
                    q = rel.get("query") or rel.get("title") or rel.get("term")
                    if q:
                        keywords.append(str(q).strip())

        trends.append(
            {
                "topic": topic[:255],
                "platform": "google_trends",
                "velocity_score": velocity,
                "summary": summary[:800],
                "keywords": keywords[:8],
                "geo": geo_code,
                "source": "apify_radar",
            }
        )
    return {
        "status": result.get("status") or "error",
        "detail": result.get("detail") or result.get("note"),
        "geo": geo_code,
        "trends": trends[:limit],
        "raw_count": len(raw_items),
        "actor": RADAR_TRENDS_ACTOR,
        "cost_cap_usd": float(RADAR_TRENDS_MAX_CHARGE_USD),
    }


async def scrape_competitor(db: AsyncSession, competitor: Competitor) -> list[CompetitorSnapshot]:
    snapshots: list[CompetitorSnapshot] = []
    agency_id = competitor.agency_id

    if competitor.website:
        web = await scrape_website(db, agency_id, competitor.website)
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="website",
                payload=web,
                summary=f"Website scan for {competitor.name}",
            )
        )
        serp = await serp_visibility(db, agency_id, competitor.name)
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="seo",
                payload=serp,
                summary=f"SEO visibility for {competitor.name}",
            )
        )

    if competitor.instagram_handle:
        ig = await run_apify_actor(
            db,
            agency_id,
            "apify~instagram-scraper",
            {"directUrls": [f"https://www.instagram.com/{competitor.instagram_handle.strip('@')}/"], "resultsLimit": 10},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="instagram",
                payload=ig,
                summary=f"Instagram activity for {competitor.name}",
            )
        )

    if competitor.tiktok_handle:
        tt = await run_apify_actor(
            db,
            agency_id,
            "clockworks~tiktok-scraper",
            {"profiles": [competitor.tiktok_handle.strip("@")], "resultsPerPage": 10},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="tiktok",
                payload=tt,
                summary=f"TikTok activity for {competitor.name}",
            )
        )

    if competitor.meta_ads_query or competitor.name:
        ads = await run_apify_actor(
            db,
            agency_id,
            "apify~facebook-ads-scraper",
            {"startUrls": [], "query": competitor.meta_ads_query or competitor.name, "maxAds": 15},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="meta_ads",
                payload=ads,
                summary=f"Meta ads for {competitor.name}",
            )
        )

    if competitor.linkedin_url:
        li = await run_apify_actor(
            db,
            agency_id,
            "harvestapi~linkedin-profile-posts",
            {"urls": [competitor.linkedin_url], "maxPosts": 10},
        )
        snapshots.append(
            CompetitorSnapshot(
                agency_id=agency_id,
                competitor_id=competitor.id,
                source="linkedin",
                payload=li,
                summary=f"LinkedIn posts for {competitor.name}",
            )
        )

    competitor.last_scraped_at = datetime.utcnow()
    for snap in snapshots:
        db.add(snap)
    return snapshots
