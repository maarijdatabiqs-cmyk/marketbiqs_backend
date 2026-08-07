from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import (
    Agency,
    BiqsTicket,
    ClientBrand,
    Competitor,
    DeliveryLog,
    FeatureComparison,
    FeatureTicket,
    GapReport,
    GoalAlert,
    InsightFeedback,
    JobStatus,
    ProductFeature,
    Report,
    TrackingJob,
)
from app.services.actions import action_run_intel
from app.services.competitive import (
    clarify_feature_descriptions,
    create_all_feature_tickets_in_jira,
    love_feature_and_build_tickets,
    run_competitive_pack,
    run_full_ai_pipeline,
)
from app.services.reports import generate_client_report

import asyncio
import logging

from fastapi.responses import JSONResponse

logger = logging.getLogger("marketbiqs.competitive.api")
router = APIRouter(tags=["competitive"])


class FeatureIn(BaseModel):
    name: str
    category: str = "General"
    description: str = ""


class FeatureOut(BaseModel):
    id: str
    name: str
    category: str
    description: str
    is_loved: bool = False
    is_wishlisted: bool = False

    model_config = {"from_attributes": True}


class WishlistIn(BaseModel):
    feature_name: str
    category: str = "General"
    description: str = ""
    competitor_id: str | None = None


class GoalsIn(BaseModel):
    goals: list[str] = Field(default_factory=list)
    niche: str | None = None
    tagline: str | None = None


def _none_to_list(value: Any) -> list:
    return value if isinstance(value, list) else []


class ComparisonOut(BaseModel):
    id: str
    competitor_id: str
    competitor_name: str
    feature_name: str
    category: str
    our_status: str
    competitor_status: str
    note: str
    how_competitor_leads: str
    how_to_improve: str
    citations: list = Field(default_factory=list)
    confidence_score: float = 0.5
    evidence_strength: str = "medium"
    feedback: str | None = None
    is_contested_move: bool = False

    model_config = {"from_attributes": True}

    @field_validator("citations", mode="before")
    @classmethod
    def citations_list(cls, value: Any) -> list:
        return _none_to_list(value)


class GapOut(BaseModel):
    id: str
    competitor_id: str
    competitor_name: str
    summary: str
    leading: list = Field(default_factory=list)
    lagging: list = Field(default_factory=list)
    opportunities: list = Field(default_factory=list)
    citations: list = Field(default_factory=list)
    confidence_score: float = 0.5
    evidence_strength: str = "medium"
    feedback: str | None = None

    model_config = {"from_attributes": True}

    @field_validator("leading", "lagging", "opportunities", "citations", mode="before")
    @classmethod
    def list_fields(cls, value: Any) -> list:
        return _none_to_list(value)


class AlertOut(BaseModel):
    id: str
    goal: str
    title: str
    why_it_matters: str
    impact: str
    action: str
    content_draft: str = ""
    estimated_cost: str = ""
    competitor_trigger: str = ""
    citations: list = Field(default_factory=list)
    confidence_score: float = 0.5
    evidence_strength: str = "medium"
    feedback: str | None = None
    acted_on: bool = False

    model_config = {"from_attributes": True}

    @field_validator("citations", mode="before")
    @classmethod
    def citations_list(cls, value: Any) -> list:
        return _none_to_list(value)


class TicketOut(BaseModel):
    id: str
    feature_id: str
    heading: str
    body: str
    acceptance_criteria: list = Field(default_factory=list)
    priority: str
    ticket_type: str
    labels: list = Field(default_factory=list)
    estimated_effort: str = ""
    story_points: int | None = None
    why_useful: str = ""
    competitor_context: str = ""
    evidence_links: list = Field(default_factory=list)
    parent_ticket_id: str | None = None
    jira_key: str | None = None
    jira_url: str | None = None
    jira_epic_key: str | None = None
    status: str

    model_config = {"from_attributes": True}

    @field_validator("acceptance_criteria", "labels", "evidence_links", mode="before")
    @classmethod
    def list_fields(cls, value: Any) -> list:
        return _none_to_list(value)


BIQS_STATUSES = ("backlog", "todo", "in_progress", "in_review", "done")
BIQS_DEFAULT_STATUS = "todo"


class BiqsTicketOut(BaseModel):
    id: str
    feature_id: str | None = None
    source_ticket_id: str | None = None
    heading: str
    body: str = ""
    acceptance_criteria: list = Field(default_factory=list)
    priority: str
    ticket_type: str
    labels: list = Field(default_factory=list)
    estimated_effort: str = ""
    story_points: int | None = None
    why_useful: str = ""
    competitor_context: str = ""
    status: str
    board_order: int = 0

    model_config = {"from_attributes": True}

    @field_validator("acceptance_criteria", "labels", mode="before")
    @classmethod
    def list_fields(cls, value: Any) -> list:
        return _none_to_list(value)


class BiqsTicketUpdate(BaseModel):
    status: str | None = None
    board_order: int | None = None

    @field_validator("status")
    @classmethod
    def known_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in BIQS_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(BIQS_STATUSES)}")
        return normalized


class FeedbackIn(BaseModel):
    entity_type: str
    entity_id: str
    rating: str
    note: str | None = None


@router.get("/clients/{client_id}/features", response_model=list[FeatureOut])
async def list_features(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ProductFeature)
        .where(ProductFeature.client_id == client.id, ProductFeature.agency_id == ctx.agency.id)
        .order_by(ProductFeature.created_at.desc())
    )
    rows = []
    for f in result.scalars().all():
        out = FeatureOut.model_validate(f)
        if f.is_loved and not f.is_wishlisted:
            out = out.model_copy(update={"is_wishlisted": True})
        rows.append(out)
    return rows


@router.post("/clients/{client_id}/features", response_model=FeatureOut)
async def add_feature(
    payload: FeatureIn,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    feature = ProductFeature(
        agency_id=ctx.agency.id,
        client_id=client.id,
        name=payload.name,
        category=payload.category,
        description=payload.description,
    )
    db.add(feature)
    await db.flush()
    return feature



@router.post("/clients/{client_id}/features/clarify", response_model=list[FeatureOut])
async def clarify_features(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    """Rewrite owned feature descriptions into plain 2–3 sentence English."""
    await clarify_feature_descriptions(db, ctx.agency, client)
    await db.flush()
    result = await db.execute(
        select(ProductFeature)
        .where(ProductFeature.client_id == client.id, ProductFeature.agency_id == ctx.agency.id)
        .order_by(ProductFeature.created_at.desc())
    )
    rows = []
    for f in result.scalars().all():
        out = FeatureOut.model_validate(f)
        if f.is_loved and not f.is_wishlisted:
            out = out.model_copy(update={"is_wishlisted": True})
        rows.append(out)
    return rows


@router.delete("/clients/{client_id}/features/{feature_id}")
async def delete_feature(
    feature_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    feature = await db.get(ProductFeature, feature_id)
    if not feature or feature.client_id != client.id or feature.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Feature not found")
    await db.delete(feature)
    await db.flush()
    return {"ok": True}


@router.patch("/clients/{client_id}/profile")
async def update_profile(
    payload: GoalsIn,
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
):
    client.goals = payload.goals
    if payload.niche is not None:
        client.niche = payload.niche
    if payload.tagline is not None:
        client.tagline = payload.tagline
    await db.flush()
    return {"id": client.id, "goals": client.goals, "niche": client.niche, "tagline": client.tagline}


@router.post("/clients/{client_id}/competitive-pack")
async def build_pack(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await run_competitive_pack(db, ctx.agency, client)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class AutoRunRequest(BaseModel):
    competitor_scope: str = Field(default="local", description="global or local")
    competitor_country: str | None = Field(default=None, max_length=120)
    competitor_count: int = Field(default=5, ge=1, le=10)
    competitor_mode: str = Field(
        default="add",
        description="update = refresh existing; add = find N new and keep previous; replace = clear auto rivals then find a fresh set",
    )

    @field_validator("competitor_scope")
    @classmethod
    def _scope(cls, value: str) -> str:
        cleaned = (value or "local").strip().lower()
        if cleaned not in {"global", "local"}:
            raise ValueError("competitor_scope must be global or local")
        return cleaned

    @field_validator("competitor_country")
    @classmethod
    def _country(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("competitor_mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        cleaned = (value or "add").strip().lower()
        if cleaned not in {"add", "update", "replace"}:
            raise ValueError("competitor_mode must be add, update, or replace")
        return cleaned


@router.post("/clients/{client_id}/auto-run")
async def auto_run(
    payload: AutoRunRequest = Body(default_factory=AutoRunRequest),
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Start intel in the background and return immediately.

    Long runs (~1–2 min) used to 500 through the Next.js/Railway rewrite proxy timeout.
    Clients should poll GET /api/clients/{id}/jobs/{job_id} (or /jobs) until completed/failed.
    """
    options = payload
    # Prefer explicit country for local; otherwise fall back to market note or global (stash UX).
    if options.competitor_scope == "local" and not options.competitor_country:
        notes = client.notes or ""
        country = ""
        for line in notes.splitlines():
            if line.lower().startswith("market:"):
                country = line.split(":", 1)[1].strip()
                break
        if country:
            options = options.model_copy(update={"competitor_country": country})
        else:
            options = options.model_copy(update={"competitor_scope": "global"})

    if ctx.agency.scrape_units_used >= ctx.agency.scrape_quota:
        raise HTTPException(status_code=402, detail="Scrape quota exceeded. Purchase client packs.")
    if ctx.agency.reports_used >= ctx.agency.reports_quota:
        raise HTTPException(status_code=402, detail="Report quota exceeded. Purchase client packs.")

    job = TrackingJob(
        agency_id=ctx.agency.id,
        client_id=client.id,
        job_type="full_ai_pipeline",
        status=JobStatus.pending,
        detail=(
            f"Queued AI pipeline ({options.competitor_mode}/{options.competitor_scope}"
            + (f" · {options.competitor_country}" if options.competitor_country else "")
            + f" · {options.competitor_count} rivals)"
        ),
        started_at=datetime.utcnow(),
        result_meta={
            "competitor_scope": options.competitor_scope,
            "competitor_country": options.competitor_country,
            "competitor_count": options.competitor_count,
            "competitor_mode": options.competitor_mode,
        },
    )
    db.add(job)
    await db.flush()
    job_id = job.id
    agency_id = ctx.agency.id
    client_id = client.id
    scope = options.competitor_scope
    country = options.competitor_country
    count = options.competitor_count
    mode = options.competitor_mode
    # Commit before background task so the row is visible to a new session
    await db.commit()

    async def _run_in_background() -> None:
        async with AsyncSessionLocal() as session:
            try:
                tracked = await session.get(TrackingJob, job_id)
                agency = await session.get(Agency, agency_id)
                brand = await session.get(ClientBrand, client_id)
                if not tracked or not agency or not brand:
                    return
                tracked.status = JobStatus.running
                tracked.detail = "Autonomous AI pipeline running"
                await session.commit()

                result = await action_run_intel(
                    session,
                    agency,
                    brand,
                    push_jira=False,
                    generate_report=True,
                    competitor_scope=scope,
                    competitor_country=country,
                    competitor_count=count,
                    competitor_mode=mode,
                )
                tracked = await session.get(TrackingJob, job_id)
                if tracked:
                    tracked.status = JobStatus.completed
                    tracked.finished_at = datetime.utcnow()
                    tracked.detail = "Autonomous AI pipeline completed"
                    tracked.result_meta = result if isinstance(result, dict) else {"ok": True}
                await session.commit()
            except Exception as exc:
                logger.exception("Background auto-run failed job=%s", job_id)
                try:
                    await session.rollback()
                    tracked = await session.get(TrackingJob, job_id)
                    if tracked:
                        tracked.status = JobStatus.failed
                        tracked.finished_at = datetime.utcnow()
                        tracked.detail = str(exc)[:800]
                        await session.commit()
                except Exception:
                    logger.exception("Failed to mark job %s as failed", job_id)

    asyncio.create_task(_run_in_background())
    return JSONResponse(
        status_code=202,
        content={
            "status": "queued",
            "job_id": job_id,
            "message": "Intel started. Poll job status until completed.",
            "competitor_scope": scope,
            "competitor_country": country,
            "competitor_count": count,
            "competitor_mode": mode,
        },
    )


@router.get("/clients/{client_id}/jobs/{job_id}")
async def get_job(
    job_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(TrackingJob, job_id)
    if not job or job.client_id != client.id or job.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "detail": job.detail,
        "result_meta": job.result_meta or {},
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }


@router.get("/clients/{client_id}/weekly-loop")
async def weekly_loop(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    client_feature_names = {
        str(f.name).lower()
        for f in (
            await db.execute(
                select(ProductFeature).where(
                    ProductFeature.client_id == client.id,
                    ProductFeature.agency_id == ctx.agency.id,
                )
            )
        ).scalars().all()
        if f.name is not None
    }

    missing_rows = (
        await db.execute(
            select(FeatureComparison)
            .where(
                FeatureComparison.client_id == client.id,
                FeatureComparison.agency_id == ctx.agency.id,
                FeatureComparison.competitor_status.in_(["leading", "strong", "has", "available"]),
                FeatureComparison.our_status.in_(["missing", "lagging", "weak", "none", "absent"]),
            )
            .order_by(FeatureComparison.confidence_score.desc())
            .limit(20)
        )
    ).scalars().all()

    missing_features = []
    seen_missing = set()
    for row in missing_rows:
        key = str(row.feature_name or "").lower()
        if not key or key in seen_missing:
            continue
        if key in client_feature_names and str(row.our_status or "") not in {"missing", "absent", "none"}:
            continue
        seen_missing.add(key)
        missing_features.append(
            {
                "type": "missing",
                "feature_name": str(row.feature_name),
                "category": str(row.category or "General"),
                "competitor_name": str(row.competitor_name),
                "competitor_id": row.competitor_id,
                "why": row.how_competitor_leads or row.note,
                "recommendation": row.how_to_improve or f"Add {row.feature_name} to close the gap vs {row.competitor_name}.",
                "confidence_score": row.confidence_score,
                "evidence_strength": row.evidence_strength,
                "citations": row.citations or [],
                "comparison_id": row.id,
            }
        )
        if len(missing_features) >= 5:
            break

    improve_rows = (
        await db.execute(
            select(FeatureComparison)
            .where(
                FeatureComparison.client_id == client.id,
                FeatureComparison.agency_id == ctx.agency.id,
                FeatureComparison.our_status.in_(["parity", "lagging", "weak"]),
                FeatureComparison.competitor_status.in_(["leading", "strong"]),
            )
            .order_by(FeatureComparison.confidence_score.desc())
            .limit(20)
        )
    ).scalars().all()
    improvements = []
    seen_improve = set()
    for row in improve_rows:
        key = str(row.feature_name or "").lower()
        if not key or key in seen_improve or key not in client_feature_names:
            continue
        if any(str(m["feature_name"]).lower() == key for m in missing_features):
            continue
        seen_improve.add(key)
        improvements.append(
            {
                "type": "improve",
                "feature_name": str(row.feature_name),
                "category": str(row.category or "General"),
                "competitor_name": str(row.competitor_name),
                "competitor_id": row.competitor_id,
                "why": row.how_competitor_leads or row.note,
                "recommendation": row.how_to_improve or f"Improve {row.feature_name} to match {row.competitor_name}.",
                "confidence_score": row.confidence_score,
                "evidence_strength": row.evidence_strength,
                "citations": row.citations or [],
                "comparison_id": row.id,
            }
        )
        if len(improvements) >= 5:
            break

    recommendations = (missing_features + improvements)[:8]

    wishlisted = (
        await db.execute(
            select(ProductFeature).where(
                ProductFeature.client_id == client.id,
                ProductFeature.agency_id == ctx.agency.id,
                or_(ProductFeature.is_wishlisted.is_(True), ProductFeature.is_loved.is_(True)),
            )
        )
    ).scalars().all()
    report = (
        await db.execute(
            select(Report)
            .where(Report.client_id == client.id, Report.agency_id == ctx.agency.id)
            .order_by(Report.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    tickets_count = (
        await db.execute(
            select(func.count())
            .select_from(FeatureTicket)
            .where(FeatureTicket.client_id == client.id, FeatureTicket.agency_id == ctx.agency.id)
        )
    ).scalar_one()

    return {
        "step": "weekly_loop",
        "recommendations": recommendations,
        "missing_features": missing_features,
        "improvements": improvements,
        "wishlisted_features": [FeatureOut.model_validate(f) for f in wishlisted],
        "loved_features": [FeatureOut.model_validate(f) for f in wishlisted],
        "latest_report": {
            "id": report.id,
            "title": report.title,
            "summary": report.summary,
            "created_at": report.created_at,
        }
        if report
        else None,
        "tickets_ready": tickets_count,
        "next_actions": [
            "Read the suggestions below — what rivals offer that this brand still lacks",
            "Save the important ones to the build list so the team can act on them",
            "Open a simple build plan (and send tasks to Jira if you use it)",
            "Write a short weekly summary you can share with the client",
        ],
    }


@router.get("/clients/{client_id}/comparisons", response_model=list[ComparisonOut])
async def list_comparisons(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    competitor_id: str | None = None,
):
    stmt = select(FeatureComparison).where(
        FeatureComparison.client_id == client.id,
        FeatureComparison.agency_id == ctx.agency.id,
    )
    if competitor_id:
        stmt = stmt.where(FeatureComparison.competitor_id == competitor_id)
    result = await db.execute(
        stmt.order_by(FeatureComparison.is_contested_move.desc(), FeatureComparison.confidence_score.desc())
    )
    return list(result.scalars().all())


@router.get("/clients/{client_id}/competitors/{competitor_id}")
async def competitor_detail(
    competitor_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    competitor = await db.get(Competitor, competitor_id)
    if not competitor or competitor.client_id != client.id or competitor.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Competitor not found")
    comparisons = (
        await db.execute(
            select(FeatureComparison)
            .where(
                FeatureComparison.client_id == client.id,
                FeatureComparison.competitor_id == competitor_id,
            )
            .order_by(FeatureComparison.confidence_score.desc())
        )
    ).scalars().all()
    gap = (
        await db.execute(
            select(GapReport)
            .where(GapReport.client_id == client.id, GapReport.competitor_id == competitor_id)
            .order_by(GapReport.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    features = competitor.feature_list or []
    if not features:
        features = [
            {
                "name": c.feature_name,
                "category": c.category,
                "description": c.how_competitor_leads or c.note or c.competitor_status,
                "status": c.competitor_status,
            }
            for c in comparisons
        ]
    return {
        "id": competitor.id,
        "name": competitor.name,
        "website": competitor.website,
        "tagline": competitor.tagline,
        "description": competitor.description or competitor.why_dangerous or "",
        "why_dangerous": competitor.why_dangerous,
        "evidence_snippet": competitor.evidence_snippet,
        "threat_level": competitor.threat_level,
        "overlap_score": competitor.overlap_score,
        "is_pinned": competitor.is_pinned,
        "is_tracking": competitor.is_tracking,
        "features": features,
        "comparisons": [ComparisonOut.model_validate(c) for c in comparisons],
        "gap": GapOut.model_validate(gap) if gap else None,
    }


@router.get("/clients/{client_id}/gaps", response_model=list[GapOut])
async def list_gaps(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(GapReport).where(GapReport.client_id == client.id, GapReport.agency_id == ctx.agency.id)
    )
    return list(result.scalars().all())


@router.get("/clients/{client_id}/alerts", response_model=list[AlertOut])
async def list_alerts(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    client_names = {
        str(f.name).lower()
        for f in (
            await db.execute(
                select(ProductFeature).where(
                    ProductFeature.client_id == client.id,
                    ProductFeature.agency_id == ctx.agency.id,
                )
            )
        ).scalars().all()
        if f.name is not None
    }
    missing_feature_names = {
        str(c.feature_name).lower()
        for c in (
            await db.execute(
                select(FeatureComparison).where(
                    FeatureComparison.client_id == client.id,
                    FeatureComparison.agency_id == ctx.agency.id,
                    FeatureComparison.our_status.in_(["missing", "lagging", "weak", "none", "absent"]),
                    FeatureComparison.competitor_status.in_(["leading", "strong", "has", "available"]),
                )
            )
        ).scalars().all()
        if c.feature_name is not None
    }
    result = await db.execute(
        select(GoalAlert)
        .where(GoalAlert.client_id == client.id, GoalAlert.agency_id == ctx.agency.id)
        .order_by(GoalAlert.created_at.desc())
    )
    alerts = []
    for alert in result.scalars().all():
        blob = f"{alert.title} {alert.why_it_matters} {alert.action} {alert.competitor_trigger}".lower()
        mentions_owned = any(n and n in blob for n in client_names)
        mentions_missing = any(n and n in blob for n in missing_feature_names)
        specialty = bool(alert.competitor_trigger) or "competitor" in blob or mentions_missing
        if specialty and (mentions_missing or not mentions_owned or "don't" in blob or "do not" in blob or "missing" in blob or "lack" in blob):
            alerts.append(alert)
        elif mentions_missing:
            alerts.append(alert)
    if not alerts:
        # fall back to high-impact alerts tied to a competitor trigger
        alerts = [
            a
            for a in (
                await db.execute(
                    select(GoalAlert)
                    .where(
                        GoalAlert.client_id == client.id,
                        GoalAlert.agency_id == ctx.agency.id,
                        GoalAlert.competitor_trigger.is_not(None),
                        GoalAlert.competitor_trigger != "",
                    )
                    .order_by(GoalAlert.created_at.desc())
                )
            ).scalars().all()
        ]
    return alerts


@router.get("/clients/{client_id}/wishlist", response_model=list[FeatureOut])
async def list_wishlist(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ProductFeature).where(
            ProductFeature.client_id == client.id,
            ProductFeature.agency_id == ctx.agency.id,
            or_(ProductFeature.is_wishlisted.is_(True), ProductFeature.is_loved.is_(True)),
        )
    )
    rows = []
    for f in result.scalars().all():
        out = FeatureOut.model_validate(f)
        rows.append(out.model_copy(update={"is_wishlisted": True}))
    return rows


@router.post("/clients/{client_id}/wishlist", response_model=FeatureOut)
async def add_to_wishlist(
    payload: WishlistIn,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    name = str(payload.feature_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="feature_name required")
    existing = (
        await db.execute(
            select(ProductFeature).where(
                ProductFeature.client_id == client.id,
                ProductFeature.agency_id == ctx.agency.id,
                func.lower(ProductFeature.name) == name.lower(),
            )
        )
    ).scalar_one_or_none()
    if existing:
        existing.is_wishlisted = True
        existing.is_loved = True
        if payload.description:
            existing.description = payload.description
        feature = existing
    else:
        feature = ProductFeature(
            agency_id=ctx.agency.id,
            client_id=client.id,
            name=name,
            category=payload.category,
            description=payload.description or f"Wishlisted from competitive intel vs rivals.",
            is_wishlisted=True,
            is_loved=True,
        )
        db.add(feature)
    await db.flush()
    return FeatureOut.model_validate(feature).model_copy(update={"is_wishlisted": True})


@router.post("/clients/{client_id}/features/{feature_id}/wishlist", response_model=FeatureOut)
async def wishlist_feature(
    feature_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    feature = await db.get(ProductFeature, feature_id)
    if not feature or feature.client_id != client.id or feature.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Feature not found")
    feature.is_wishlisted = True
    feature.is_loved = True
    await db.flush()
    return FeatureOut.model_validate(feature).model_copy(update={"is_wishlisted": True})


@router.post("/clients/{client_id}/features/{feature_id}/development-plan", response_model=list[TicketOut])
async def development_plan(
    feature_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    feature = await db.get(ProductFeature, feature_id)
    if not feature or feature.client_id != client.id or feature.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Feature not found")
    feature.is_wishlisted = True
    feature.is_loved = True
    # Generate draft tickets only — Jira push is a separate explicit action to avoid proxy timeouts.
    tickets = await love_feature_and_build_tickets(db, ctx.agency, client, feature)
    return tickets


@router.post("/clients/{client_id}/feedback")
async def submit_feedback(
    payload: FeedbackIn,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    if payload.rating not in {"useful", "useless"}:
        raise HTTPException(status_code=400, detail="rating must be useful or useless")
    if payload.entity_type not in {"comparison", "gap", "alert"}:
        raise HTTPException(status_code=400, detail="Invalid entity_type")

    entity = None
    if payload.entity_type == "comparison":
        entity = await db.get(FeatureComparison, payload.entity_id)
    elif payload.entity_type == "gap":
        entity = await db.get(GapReport, payload.entity_id)
    else:
        entity = await db.get(GoalAlert, payload.entity_id)
    if not entity or getattr(entity, "client_id", None) != client.id:
        raise HTTPException(status_code=404, detail="Entity not found")
    entity.feedback = payload.rating
    db.add(
        InsightFeedback(
            agency_id=ctx.agency.id,
            client_id=client.id,
            entity_type=payload.entity_type,
            entity_id=payload.entity_id,
            rating=payload.rating,
            note=payload.note,
        )
    )
    await db.flush()
    return {"ok": True, "feedback": payload.rating}


@router.post("/clients/{client_id}/alerts/{alert_id}/acted")
async def mark_alert_acted(
    alert_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    alert = await db.get(GoalAlert, alert_id)
    if not alert or alert.client_id != client.id or alert.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acted_on = True
    await db.flush()
    return {"ok": True}


@router.post("/clients/{client_id}/competitors/{competitor_id}/pin")
async def pin_competitor(
    competitor_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    competitor = await db.get(Competitor, competitor_id)
    if not competitor or competitor.client_id != client.id or competitor.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Competitor not found")
    competitor.is_pinned = True
    competitor.is_tracking = True
    await db.flush()
    return {"ok": True, "is_pinned": True}


@router.post("/clients/{client_id}/competitors/{competitor_id}/unpin")
async def unpin_competitor(
    competitor_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    competitor = await db.get(Competitor, competitor_id)
    if not competitor or competitor.client_id != client.id or competitor.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Competitor not found")
    competitor.is_pinned = False
    await db.flush()
    return {"ok": True, "is_pinned": False}


@router.post("/clients/{client_id}/features/{feature_id}/love", response_model=list[TicketOut])
async def love_feature(
    feature_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    feature = await db.get(ProductFeature, feature_id)
    if not feature or feature.client_id != client.id or feature.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Feature not found")
    return await love_feature_and_build_tickets(db, ctx.agency, client, feature)


@router.get("/clients/{client_id}/features/{feature_id}/tickets", response_model=list[TicketOut])
async def list_feature_tickets(
    feature_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(FeatureTicket)
        .where(
            FeatureTicket.feature_id == feature_id,
            FeatureTicket.client_id == client.id,
            FeatureTicket.agency_id == ctx.agency.id,
        )
        .order_by(FeatureTicket.created_at.asc())
    )
    return list(result.scalars().all())


@router.post("/clients/{client_id}/features/{feature_id}/tickets/create-all", response_model=list[TicketOut])
async def create_all_tickets(
    feature_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await create_all_feature_tickets_in_jira(db, ctx.agency.id, client.id, feature_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/clients/{client_id}/features/{feature_id}/tickets/push-biqs", response_model=list[BiqsTicketOut])
async def push_tickets_to_biqs(
    feature_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    source = (
        await db.execute(
            select(FeatureTicket)
            .where(
                FeatureTicket.feature_id == feature_id,
                FeatureTicket.client_id == client.id,
                FeatureTicket.agency_id == ctx.agency.id,
            )
            .order_by(FeatureTicket.created_at.asc())
        )
    ).scalars().all()
    if not source:
        raise HTTPException(status_code=400, detail="No tickets found. Generate a development plan first.")

    already = set(
        (
            await db.execute(
                select(BiqsTicket.source_ticket_id).where(
                    BiqsTicket.client_id == client.id,
                    BiqsTicket.agency_id == ctx.agency.id,
                    BiqsTicket.source_ticket_id.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    start_order = (
        await db.execute(
            select(func.count())
            .select_from(BiqsTicket)
            .where(BiqsTicket.client_id == client.id, BiqsTicket.status == BIQS_DEFAULT_STATUS)
        )
    ).scalar_one()

    created: list[BiqsTicket] = []
    for offset, ticket in enumerate(t for t in source if t.id not in already):
        card = BiqsTicket(
            agency_id=ctx.agency.id,
            client_id=client.id,
            feature_id=ticket.feature_id,
            source_ticket_id=ticket.id,
            heading=ticket.heading,
            body=ticket.body or "",
            acceptance_criteria=ticket.acceptance_criteria or [],
            priority=ticket.priority,
            ticket_type=ticket.ticket_type,
            labels=ticket.labels or [],
            estimated_effort=ticket.estimated_effort or "",
            story_points=ticket.story_points,
            why_useful=ticket.why_useful or "",
            competitor_context=ticket.competitor_context or "",
            status=BIQS_DEFAULT_STATUS,
            board_order=start_order + offset,
        )
        db.add(card)
        created.append(card)
    await db.flush()
    return created


@router.get("/clients/{client_id}/biqs-tickets", response_model=list[BiqsTicketOut])
async def list_biqs_tickets(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BiqsTicket)
        .where(BiqsTicket.client_id == client.id, BiqsTicket.agency_id == ctx.agency.id)
        .order_by(BiqsTicket.board_order.asc(), BiqsTicket.created_at.asc())
    )
    return list(result.scalars().all())


@router.patch("/clients/{client_id}/biqs-tickets/{ticket_id}", response_model=BiqsTicketOut)
async def update_biqs_ticket(
    ticket_id: str,
    payload: BiqsTicketUpdate,
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    ticket = await db.get(BiqsTicket, ticket_id)
    if not ticket or ticket.client_id != client.id or ticket.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if payload.status is not None:
        ticket.status = payload.status
    if payload.board_order is not None:
        ticket.board_order = payload.board_order
    await db.flush()
    return ticket


@router.post("/clients/{client_id}/weekly-brief")
async def weekly_brief(
    client: ClientBrand = Depends(get_tenant_client),
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    report = await generate_client_report(db, ctx.agency, client, period_label="Weekly Loop Brief")
    return {"id": report.id, "title": report.title, "summary": report.summary}


@router.get("/agency/roi")
async def agency_roi(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    agency_id = ctx.agency.id
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    reports_month = (
        await db.execute(
            select(func.count())
            .select_from(Report)
            .where(Report.agency_id == agency_id, Report.created_at >= month_start)
        )
    ).scalar_one()
    tickets_month = (
        await db.execute(
            select(func.count())
            .select_from(FeatureTicket)
            .where(FeatureTicket.agency_id == agency_id, FeatureTicket.created_at >= month_start)
        )
    ).scalar_one()
    jira_month = (
        await db.execute(
            select(func.count())
            .select_from(FeatureTicket)
            .where(
                FeatureTicket.agency_id == agency_id,
                FeatureTicket.created_at >= month_start,
                FeatureTicket.jira_key.is_not(None),
            )
        )
    ).scalar_one()
    alerts_acted = (
        await db.execute(
            select(func.count())
            .select_from(GoalAlert)
            .where(GoalAlert.agency_id == agency_id, GoalAlert.acted_on.is_(True), GoalAlert.created_at >= month_start)
        )
    ).scalar_one()
    deliveries = (
        await db.execute(
            select(func.count())
            .select_from(DeliveryLog)
            .where(DeliveryLog.agency_id == agency_id, DeliveryLog.created_at >= month_start)
        )
    ).scalar_one()
    useful = (
        await db.execute(
            select(func.count())
            .select_from(InsightFeedback)
            .where(InsightFeedback.agency_id == agency_id, InsightFeedback.rating == "useful")
        )
    ).scalar_one()
    useless = (
        await db.execute(
            select(func.count())
            .select_from(InsightFeedback)
            .where(InsightFeedback.agency_id == agency_id, InsightFeedback.rating == "useless")
        )
    ).scalar_one()

    hours_saved = round(
        reports_month * 2.5 + tickets_month * 0.35 + alerts_acted * 0.5 + ctx.agency.scrape_units_used * 0.01,
        1,
    )
    return {
        "month_start": month_start,
        "hours_saved_estimate": hours_saved,
        "alerts_acted_on": alerts_acted,
        "tickets_created": tickets_month,
        "jira_tickets_pushed": jira_month,
        "reports_delivered": reports_month,
        "deliveries_sent": deliveries,
        "feedback_useful": useful,
        "feedback_useless": useless,
        "scrape_units_used": ctx.agency.scrape_units_used,
        "scrape_quota": ctx.agency.scrape_quota,
        "reports_used": ctx.agency.reports_used,
        "reports_quota": ctx.agency.reports_quota,
    }
