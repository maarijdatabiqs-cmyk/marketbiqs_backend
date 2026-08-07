from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import AuthContext, get_auth_context, get_tenant_client
from app.models import ClientBrand, Competitor, FeatureTicket, GoalAlert, ProductFeature, Report
from app.schemas import ClientCreate, ClientOut, ClientUpdate, CompetitorCreate, CompetitorOut
from app.services.billing import ensure_client_capacity

router = APIRouter(prefix="/clients", tags=["clients"])


async def _enrich_client(db: AsyncSession, client: ClientBrand) -> ClientOut:
    rivals = (
        await db.execute(
            select(func.count())
            .select_from(Competitor)
            .where(Competitor.client_id == client.id, Competitor.is_tracking.is_(True))
        )
    ).scalar_one()
    features = (
        await db.execute(select(func.count()).select_from(ProductFeature).where(ProductFeature.client_id == client.id))
    ).scalar_one()
    reports = (
        await db.execute(select(func.count()).select_from(Report).where(Report.client_id == client.id))
    ).scalar_one()
    tickets = (
        await db.execute(select(func.count()).select_from(FeatureTicket).where(FeatureTicket.client_id == client.id))
    ).scalar_one()
    alerts_open = (
        await db.execute(
            select(func.count())
            .select_from(GoalAlert)
            .where(GoalAlert.client_id == client.id, GoalAlert.acted_on.is_(False))
        )
    ).scalar_one()
    data = ClientOut.model_validate(client)
    return data.model_copy(
        update={
            "rivals_count": rivals,
            "features_count": features,
            "reports_count": reports,
            "tickets_count": tickets,
            "alerts_open": alerts_open,
        }
    )


@router.get("", response_model=list[ClientOut])
async def list_clients(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    include_inactive: bool = False,
):
    stmt = select(ClientBrand).where(ClientBrand.agency_id == ctx.agency.id)
    if not include_inactive:
        stmt = stmt.where(ClientBrand.is_active.is_(True))
    result = await db.execute(stmt.order_by(ClientBrand.created_at.desc()))
    clients = list(result.scalars().all())
    return [await _enrich_client(db, c) for c in clients]


@router.post("", response_model=ClientOut)
async def create_client(
    payload: ClientCreate,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    try:
        await ensure_client_capacity(db, ctx.agency)
    except ValueError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    client = ClientBrand(agency_id=ctx.agency.id, **payload.model_dump())
    db.add(client)
    await db.flush()
    # Intel is started explicitly by the UI via POST /clients/{id}/auto-run
    # (background create-time runs raced the UI and often finished with no feedback).
    return await _enrich_client(db, client)


@router.get("/{client_id}", response_model=ClientOut)
async def get_client(client: ClientBrand = Depends(get_tenant_client), db: AsyncSession = Depends(get_db)):
    return await _enrich_client(db, client)


@router.patch("/{client_id}", response_model=ClientOut)
async def update_client(
    payload: ClientUpdate,
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
):
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(client, key, value)
    await db.flush()
    return await _enrich_client(db, client)


@router.delete("/{client_id}")
async def delete_client(
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
):
    client.is_active = False
    await db.flush()
    return {"ok": True}


@router.get("/{client_id}/competitors", response_model=list[CompetitorOut])
async def list_competitors(
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = Depends(get_auth_context),
    include_hidden: bool = False,
):
    stmt = select(Competitor).where(
        Competitor.client_id == client.id,
        Competitor.agency_id == ctx.agency.id,
    )
    if not include_hidden:
        stmt = stmt.where(Competitor.is_tracking.is_(True))
    result = await db.execute(
        stmt.order_by(Competitor.is_pinned.desc(), Competitor.is_tracking.desc(), Competitor.overlap_score.desc())
    )
    return list(result.scalars().all())


@router.post("/{client_id}/competitors", response_model=CompetitorOut)
async def add_competitor(
    payload: CompetitorCreate,
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = Depends(get_auth_context),
):
    data = payload.model_dump()
    competitor = Competitor(
        agency_id=ctx.agency.id,
        client_id=client.id,
        **data,
        # Keep manual rivals in intel runs (protected from AI prune + count slice)
        is_tracking=True,
        is_pinned=True,
        overlap_score=75.0,
        threat_level="high",
        why_dangerous=f"Manually added rival for {client.name}",
    )
    db.add(competitor)
    await db.flush()
    await db.refresh(competitor)
    return competitor


@router.delete("/{client_id}/competitors/{competitor_id}")
async def remove_competitor(
    competitor_id: str,
    client: ClientBrand = Depends(get_tenant_client),
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = Depends(get_auth_context),
):
    competitor = await db.get(Competitor, competitor_id)
    if not competitor or competitor.client_id != client.id or competitor.agency_id != ctx.agency.id:
        raise HTTPException(status_code=404, detail="Competitor not found")
    await db.delete(competitor)
    await db.flush()
    return {"ok": True}
