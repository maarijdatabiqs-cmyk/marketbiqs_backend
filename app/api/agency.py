from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.deps import AuthContext, get_auth_context
from app.models import (
    AgencyMember,
    ClientBrand,
    Competitor,
    DeliveryLog,
    FeatureComparison,
    FeatureTicket,
    GapReport,
    GoalAlert,
    Insight,
    InsightFeedback,
    MemberRole,
    ProductFeature,
    Report,
    TrackingJob,
    TrendSignal,
    UsageEvent,
    User,
)
from app.schemas import (
    AgencyBrandUpdate,
    AgencyOut,
    DashboardOut,
    MemberInvite,
    MemberOut,
)
from app.config import get_settings
from app.services.billing import compute_budget

router = APIRouter(prefix="/agency", tags=["agency"])
settings = get_settings()


async def _invite_supabase_user(email: str, full_name: str) -> dict:
    """Invite teammate via Supabase Auth (email magic invite). Returns Auth user payload."""
    import httpx

    base = (settings.supabase_url or "").strip().rstrip("/")
    secret = settings.resolved_secret_key()
    if not base or not secret:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured for invites. Set SUPABASE_URL and SUPABASE_SECRET_KEY.",
        )
    headers = {
        "apikey": secret,
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    redirect_to = f"{(settings.frontend_url or '').rstrip('/')}/auth/callback" if settings.frontend_url else None
    body: dict = {
        "email": email.lower(),
        "data": {"full_name": full_name},
    }
    if redirect_to:
        body["redirect_to"] = redirect_to
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prefer invite endpoint (sends email). Fall back to admin create + generate link.
        res = await client.post(f"{base}/auth/v1/invite", headers=headers, json=body)
        if res.status_code >= 400:
            # Existing Auth user: look them up so we can still attach membership
            listed = await client.get(
                f"{base}/auth/v1/admin/users",
                headers=headers,
                params={"page": 1, "per_page": 200},
            )
            if listed.status_code == 200:
                users = (listed.json() or {}).get("users") or []
                match = next((u for u in users if (u.get("email") or "").lower() == email.lower()), None)
                if match:
                    return match
            detail = res.text
            try:
                payload = res.json()
                detail = (
                    payload.get("msg")
                    or payload.get("error_description")
                    or payload.get("message")
                    or payload.get("error")
                    or detail
                )
            except Exception:
                pass
            if res.status_code == 429 or "rate limit" in str(detail).lower():
                raise HTTPException(
                    status_code=429,
                    detail="Invite email rate limit hit (Supabase free email ~2/hour). Wait and try once, or connect custom SMTP.",
                )
            raise HTTPException(status_code=400, detail=f"Invite failed: {detail}")
        return res.json()


@router.get("/dashboard", response_model=DashboardOut)
async def dashboard(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    agency_id = ctx.agency.id
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    clients_count = (
        await db.execute(select(func.count()).select_from(ClientBrand).where(ClientBrand.agency_id == agency_id))
    ).scalar_one()
    competitors_count = (
        await db.execute(
            select(func.count())
            .select_from(Competitor)
            .where(Competitor.agency_id == agency_id, Competitor.is_tracking.is_(True))
        )
    ).scalar_one()
    reports_count = (
        await db.execute(select(func.count()).select_from(Report).where(Report.agency_id == agency_id))
    ).scalar_one()
    open_insights = (
        await db.execute(select(func.count()).select_from(Insight).where(Insight.agency_id == agency_id))
    ).scalar_one()
    recent_trends = (
        await db.execute(
            select(TrendSignal).where(TrendSignal.agency_id == agency_id).order_by(TrendSignal.detected_at.desc()).limit(6)
        )
    ).scalars().all()
    recent_insights = (
        await db.execute(
            select(Insight).where(Insight.agency_id == agency_id).order_by(Insight.created_at.desc()).limit(6)
        )
    ).scalars().all()
    active_clients = (
        await db.execute(
            select(func.count())
            .select_from(ClientBrand)
            .where(ClientBrand.agency_id == agency_id, ClientBrand.is_active.is_(True))
        )
    ).scalar_one()
    reports_month = (
        await db.execute(
            select(func.count()).select_from(Report).where(Report.agency_id == agency_id, Report.created_at >= month_start)
        )
    ).scalar_one()
    tickets_month = (
        await db.execute(
            select(func.count())
            .select_from(FeatureTicket)
            .where(FeatureTicket.agency_id == agency_id, FeatureTicket.created_at >= month_start)
        )
    ).scalar_one()
    alerts_acted = (
        await db.execute(
            select(func.count())
            .select_from(GoalAlert)
            .where(GoalAlert.agency_id == agency_id, GoalAlert.acted_on.is_(True), GoalAlert.created_at >= month_start)
        )
    ).scalar_one()
    alerts_total = (
        await db.execute(
            select(func.count()).select_from(GoalAlert).where(GoalAlert.agency_id == agency_id, GoalAlert.created_at >= month_start)
        )
    ).scalar_one()
    deliveries = (
        await db.execute(
            select(func.count())
            .select_from(DeliveryLog)
            .where(DeliveryLog.agency_id == agency_id, DeliveryLog.created_at >= month_start)
        )
    ).scalar_one()
    jira_pushed = (
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

    threat_rows = (
        await db.execute(
            select(Competitor.threat_level, func.count())
            .where(Competitor.agency_id == agency_id, Competitor.is_tracking.is_(True))
            .group_by(Competitor.threat_level)
        )
    ).all()
    threat_breakdown = [{"label": (lvl or "unknown"), "value": int(cnt)} for lvl, cnt in threat_rows]

    delivery_status_rows = (
        await db.execute(
            select(DeliveryLog.status, func.count())
            .where(DeliveryLog.agency_id == agency_id, DeliveryLog.created_at >= month_start)
            .group_by(DeliveryLog.status)
        )
    ).all()
    delivery_breakdown = [{"label": (st or "unknown"), "value": int(cnt)} for st, cnt in delivery_status_rows]

    # Day-bucketed activity using Python side grouping for SQLite compatibility
    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=13)

    async def series_for(model, col):
        items = (
            await db.execute(select(col).where(model.agency_id == agency_id, col >= start))
        ).scalars().all()
        bucket: dict[str, int] = {}
        for dt in items:
            if not dt:
                continue
            key = dt.strftime("%Y-%m-%d")
            bucket[key] = bucket.get(key, 0) + 1
        out = []
        for i in range(14):
            day = start + timedelta(days=i)
            key = day.strftime("%Y-%m-%d")
            out.append({"date": key, "label": day.strftime("%b %d"), "value": bucket.get(key, 0)})
        return out

    reports_series = await series_for(Report, Report.created_at)
    tickets_series = await series_for(FeatureTicket, FeatureTicket.created_at)
    alerts_series = await series_for(GoalAlert, GoalAlert.created_at)
    deliveries_series = await series_for(DeliveryLog, DeliveryLog.created_at)
    scrapes_series = await series_for(UsageEvent, UsageEvent.created_at)

    activity = []
    for i in range(14):
        activity.append(
            {
                "date": reports_series[i]["date"],
                "label": reports_series[i]["label"],
                "reports": reports_series[i]["value"],
                "tickets": tickets_series[i]["value"],
                "alerts": alerts_series[i]["value"],
                "deliveries": deliveries_series[i]["value"],
                "scrapes": scrapes_series[i]["value"],
            }
        )

    clients = (
        await db.execute(
            select(ClientBrand)
            .where(ClientBrand.agency_id == agency_id, ClientBrand.is_active.is_(True))
            .order_by(ClientBrand.created_at.desc())
        )
    ).scalars().all()
    portfolio = []
    for client in clients:
        rivals = (
            await db.execute(
                select(func.count())
                .select_from(Competitor)
                .where(Competitor.client_id == client.id, Competitor.is_tracking.is_(True))
            )
        ).scalar_one()
        feats = (
            await db.execute(select(func.count()).select_from(ProductFeature).where(ProductFeature.client_id == client.id))
        ).scalar_one()
        reps = (
            await db.execute(select(func.count()).select_from(Report).where(Report.client_id == client.id))
        ).scalar_one()
        tix = (
            await db.execute(select(func.count()).select_from(FeatureTicket).where(FeatureTicket.client_id == client.id))
        ).scalar_one()
        gaps = (
            await db.execute(select(func.count()).select_from(GapReport).where(GapReport.client_id == client.id))
        ).scalar_one()
        alerts = (
            await db.execute(select(func.count()).select_from(GoalAlert).where(GoalAlert.client_id == client.id))
        ).scalar_one()
        wishlist = (
            await db.execute(
                select(func.count())
                .select_from(ProductFeature)
                .where(
                    ProductFeature.client_id == client.id,
                    ProductFeature.is_wishlisted.is_(True),
                )
            )
        ).scalar_one()
        last_job_at = (
            await db.execute(
                select(TrackingJob.finished_at)
                .where(
                    TrackingJob.client_id == client.id,
                    TrackingJob.agency_id == agency_id,
                    TrackingJob.finished_at.is_not(None),
                )
                .order_by(TrackingJob.finished_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        last_report_at = (
            await db.execute(
                select(Report.created_at)
                .where(Report.client_id == client.id, Report.agency_id == agency_id)
                .order_by(Report.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        candidates = [dt for dt in (last_job_at, last_report_at) if dt is not None]
        last_intel_at = max(candidates) if candidates else None
        portfolio.append(
            {
                "id": client.id,
                "name": client.name,
                "industry": client.industry,
                "is_active": client.is_active,
                "rivals": rivals,
                "features": feats,
                "reports": reps,
                "tickets": tix,
                "gaps": gaps,
                "alerts": alerts,
                "wishlist": wishlist,
                "last_intel_at": last_intel_at.isoformat() if last_intel_at else None,
            }
        )

    owned_features = (
        await db.execute(
            select(func.count())
            .select_from(ProductFeature)
            .where(
                ProductFeature.agency_id == agency_id,
                ProductFeature.is_wishlisted.is_(False),
            )
        )
    ).scalar_one()
    wishlist_features = (
        await db.execute(
            select(func.count())
            .select_from(ProductFeature)
            .where(
                ProductFeature.agency_id == agency_id,
                ProductFeature.is_wishlisted.is_(True),
            )
        )
    ).scalar_one()

    alert_impact_rows = (
        await db.execute(
            select(GoalAlert.impact, func.count())
            .where(GoalAlert.agency_id == agency_id)
            .group_by(GoalAlert.impact)
        )
    ).all()
    impact_counts = {"high": 0, "medium": 0, "low": 0}
    for lvl, cnt in alert_impact_rows:
        key = str(lvl or "medium").strip().lower()
        if key not in impact_counts:
            # Bad AI text historically landed in impact — bucket as medium
            key = "medium" if len(key) > 12 else key
            if key not in impact_counts:
                key = "medium"
        impact_counts[key] += int(cnt)
    alert_impact = [{"label": k, "value": v} for k, v in impact_counts.items() if v > 0]
    if not alert_impact:
        total_alerts = (
            await db.execute(select(func.count()).select_from(GoalAlert).where(GoalAlert.agency_id == agency_id))
        ).scalar_one()
        if total_alerts:
            alert_impact = [{"label": "open", "value": int(total_alerts)}]

    comparison_rows = (
        await db.execute(
            select(FeatureComparison.our_status, func.count())
            .where(FeatureComparison.agency_id == agency_id)
            .group_by(FeatureComparison.our_status)
        )
    ).all()
    posture_map = {"leading": 0, "parity": 0, "lagging": 0}
    for st, cnt in comparison_rows:
        key = str(st or "parity").strip().lower()
        if key not in posture_map:
            key = "parity"
        posture_map[key] += int(cnt)
    comparison_posture = [{"label": k, "value": v} for k, v in posture_map.items() if v > 0]

    overlap_scores = (
        await db.execute(
            select(Competitor.overlap_score).where(
                Competitor.agency_id == agency_id,
                Competitor.is_tracking.is_(True),
            )
        )
    ).scalars().all()
    overlap_buckets = [
        {"label": "0-25%", "value": 0},
        {"label": "26-50%", "value": 0},
        {"label": "51-75%", "value": 0},
        {"label": "76-100%", "value": 0},
    ]
    for score in overlap_scores:
        try:
            value = float(score or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 25:
            overlap_buckets[0]["value"] += 1
        elif value <= 50:
            overlap_buckets[1]["value"] += 1
        elif value <= 75:
            overlap_buckets[2]["value"] += 1
        else:
            overlap_buckets[3]["value"] += 1
    if sum(b["value"] for b in overlap_buckets) == 0:
        # Still show structure when rivals exist but overlap scores are empty
        rival_count = (
            await db.execute(
                select(func.count())
                .select_from(Competitor)
                .where(Competitor.agency_id == agency_id, Competitor.is_tracking.is_(True))
            )
        ).scalar_one()
        if rival_count:
            overlap_buckets[2]["value"] = int(rival_count)

    industry_counts: dict[str, int] = {}
    for client in clients:
        label = (client.industry or "Unspecified").strip() or "Unspecified"
        industry_counts[label] = industry_counts.get(label, 0) + 1
    industry_mix = [{"label": k, "value": v} for k, v in sorted(industry_counts.items(), key=lambda x: -x[1])]

    if not delivery_breakdown:
        # Fall back to report/delivery volume so the chart isn't blank forever
        sent = (
            await db.execute(select(func.count()).select_from(DeliveryLog).where(DeliveryLog.agency_id == agency_id))
        ).scalar_one()
        if sent:
            delivery_breakdown = [{"label": "logged", "value": int(sent)}]
        elif reports_count:
            delivery_breakdown = [
                {"label": "reports ready", "value": int(reports_count)},
                {"label": "delivered", "value": 0},
            ]

    top_clients = portfolio[:8]
    client_rivals = [{"label": c["name"][:18], "value": c["rivals"]} for c in top_clients]
    client_gaps = [{"label": c["name"][:18], "value": c["gaps"]} for c in top_clients]
    client_alerts = [{"label": c["name"][:18], "value": c["alerts"]} for c in top_clients]
    client_coverage = [
        {
            "label": c["name"][:14],
            "rivals": c["rivals"],
            "features": c["features"],
            "wishlist": c["wishlist"],
            "tickets": c["tickets"],
        }
        for c in top_clients
    ]
    scrape_activity = [{"label": d["label"], "value": d["scrapes"]} for d in activity]

    hours_saved = round(reports_month * 2.5 + tickets_month * 0.35 + alerts_acted * 0.5 + jira_pushed * 0.4, 1)
    usage = compute_budget(ctx.agency, active_clients)

    return DashboardOut(
        agency=AgencyOut.model_validate(ctx.agency),
        clients_count=clients_count,
        competitors_count=competitors_count,
        reports_count=reports_count,
        open_insights=open_insights,
        recent_trends=recent_trends,
        recent_insights=recent_insights,
        usage=usage,
        roi={
            "hours_saved_estimate": hours_saved,
            "alerts_acted_on": alerts_acted,
            "alerts_total": alerts_total,
            "tickets_created": tickets_month,
            "jira_tickets_pushed": jira_pushed,
            "reports_delivered": reports_month,
            "deliveries_sent": deliveries,
            "feedback_useful": useful,
            "feedback_useless": useless,
        },
        charts={
            "activity": activity,
            "scrape_activity": scrape_activity,
            "threat_breakdown": threat_breakdown,
            "delivery_breakdown": delivery_breakdown,
            "feature_split": [
                {"label": "Owned", "value": int(owned_features)},
                {"label": "Wishlist", "value": int(wishlist_features)},
            ],
            "alert_impact": alert_impact,
            "comparison_posture": comparison_posture,
            "overlap_buckets": overlap_buckets,
            "industry_mix": industry_mix,
            "client_rivals": client_rivals,
            "client_gaps": client_gaps,
            "client_alerts": client_alerts,
            "client_coverage": client_coverage,
            "usage_bars": [
                {
                    "label": "Clients",
                    "used": usage.active_clients,
                    "quota": usage.max_clients,
                },
                {
                    "label": "Reports",
                    "used": usage.reports_used,
                    "quota": usage.reports_quota,
                },
                {
                    "label": "Scrapes",
                    "used": usage.scrape_units_used,
                    "quota": usage.scrape_quota,
                },
            ],
            "roi_bars": [
                {"label": "Hours saved", "value": hours_saved},
                {"label": "Reports", "value": reports_month},
                {"label": "Tickets", "value": tickets_month},
                {"label": "Jira pushed", "value": jira_pushed},
                {"label": "Alerts acted", "value": alerts_acted},
                {"label": "Deliveries", "value": deliveries},
            ],
            "feedback": [
                {"label": "Useful", "value": useful},
                {"label": "Useless", "value": useless},
            ],
        },
        portfolio=portfolio,
    )


def _normalize_brand_hex(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if not raw.startswith("#"):
        raw = f"#{raw}"
    if len(raw) == 4 and all(c in "0123456789abcdefABCDEF" for c in raw[1:]):
        raw = f"#{raw[1]*2}{raw[2]*2}{raw[3]*2}"
    if len(raw) != 7 or any(c not in "0123456789abcdefABCDEF#" for c in raw):
        raise HTTPException(status_code=400, detail=f"Invalid brand color: {value}")
    return raw.lower()


@router.patch("/branding", response_model=AgencyOut)
async def update_branding(
    payload: AgencyBrandUpdate,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    role = ctx.membership.role.value if hasattr(ctx.membership.role, "value") else str(ctx.membership.role)
    if role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners/admins can update branding")
    agency = ctx.agency
    data = payload.model_dump(exclude_unset=True)
    if "brand_color" in data:
        data["brand_color"] = _normalize_brand_hex(data["brand_color"]) or agency.brand_color
    if "brand_secondary" in data:
        data["brand_secondary"] = _normalize_brand_hex(data["brand_secondary"]) or agency.brand_secondary
    if "logo_url" in data and isinstance(data["logo_url"], str):
        data["logo_url"] = data["logo_url"].strip() or None
    if "report_footer" in data and isinstance(data["report_footer"], str):
        data["report_footer"] = data["report_footer"].strip() or None
    for key, value in data.items():
        setattr(agency, key, value)
    agency.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(agency)
    return AgencyOut.model_validate(agency)


@router.get("/members", response_model=list[MemberOut])
async def list_members(ctx: AuthContext = Depends(get_auth_context), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AgencyMember)
        .options(selectinload(AgencyMember.user))
        .where(AgencyMember.agency_id == ctx.agency.id, AgencyMember.is_active.is_(True))
    )
    return list(result.scalars().all())


@router.post("/members", response_model=MemberOut)
async def invite_member(
    payload: MemberInvite,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    if ctx.membership.role not in (MemberRole.owner, MemberRole.admin):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    try:
        role = MemberRole(payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid role") from exc

    email = payload.email.lower().strip()
    auth_user = await _invite_supabase_user(email, payload.full_name.strip())
    user_id = auth_user.get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Invite did not return a user id")

    user = await db.get(User, user_id)
    if not user:
        # Claim legacy email row if present (pre-Supabase invite leftovers)
        legacy = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if legacy and legacy.id != user_id:
            memberships = (
                await db.execute(select(AgencyMember).where(AgencyMember.user_id == legacy.id))
            ).scalars().all()
            for membership in memberships:
                membership.user_id = user_id
            await db.delete(legacy)
            await db.flush()
        user = User(
            id=user_id,
            email=email,
            full_name=payload.full_name.strip()[:255] or email.split("@")[0],
            hashed_password=None,
        )
        db.add(user)
        await db.flush()
    else:
        if payload.full_name.strip():
            user.full_name = payload.full_name.strip()[:255]

    existing = (
        await db.execute(
            select(AgencyMember).where(
                AgencyMember.agency_id == ctx.agency.id,
                AgencyMember.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        existing.is_active = True
        existing.role = role
        existing.invited_email = email
        member = existing
    else:
        member = AgencyMember(
            agency_id=ctx.agency.id,
            user_id=user.id,
            role=role,
            invited_email=email,
        )
        db.add(member)
    await db.flush()
    await db.refresh(member, attribute_names=["user"])
    return member
