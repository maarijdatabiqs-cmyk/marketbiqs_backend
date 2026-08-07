from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import logging
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from croniter import croniter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.api import agency, auth, billing, chat, clients, competitive, delivery, intelligence, integrations, reports, supabase_api, whitelabel
from app.config import get_settings
from app.database import DATABASE_BACKEND, AsyncSessionLocal, init_db, ping_db
from app.models import Agency, ClientBrand, Report
from app.services.actions import action_deliver, action_run_intel
from app.services.supabase_client import ensure_reports_bucket, ping_supabase, supabase_configured

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("marketbiqs")
# APScheduler otherwise floods Railway with "Running job … executed successfully" every minute
logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)

settings = get_settings()
scheduler = AsyncIOScheduler()
_startup_ok = {"db": False, "error": None}


async def scheduled_ai_pipeline() -> None:
    async with AsyncSessionLocal() as db:
        clients = (await db.execute(select(ClientBrand).where(ClientBrand.is_active.is_(True)))).scalars().all()
        for client in clients:
            agency = await db.get(Agency, client.agency_id)
            if not agency:
                continue
            try:
                await action_run_intel(db, agency, client, push_jira=True, generate_report=True)
                await db.commit()
            except Exception:
                logger.exception("Scheduled intel failed for client %s", client.id)
                await db.rollback()


async def scheduled_delivery_pipeline() -> None:
    """Send due client deliveries based on delivery_schedule_cron.

    Runs often so cron minutes are not missed; quiet unless a delivery is actually sent.
    """
    now = datetime.utcnow()
    sent = 0
    async with AsyncSessionLocal() as db:
        clients = (
            await db.execute(select(ClientBrand).where(ClientBrand.is_active.is_(True)))
        ).scalars().all()
        for client in clients:
            cron_expr = (client.delivery_schedule_cron or "").strip()
            if not cron_expr:
                continue
            try:
                base = now.replace(second=0, microsecond=0)
                itr = croniter(cron_expr, base)
                prev = itr.get_prev(datetime)
                if prev != base:
                    continue
            except Exception:
                continue
            agency = await db.get(Agency, client.agency_id)
            if not agency:
                continue
            report = (
                await db.execute(
                    select(Report)
                    .where(Report.client_id == client.id)
                    .order_by(Report.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            try:
                await action_deliver(db, agency, client, report=report, message=None)
                await db.commit()
                sent += 1
                logger.info("Scheduled delivery sent for client=%s agency=%s", client.id, agency.id)
            except Exception:
                logger.exception("Scheduled delivery failed for client %s", client.id)
                await db.rollback()
    if sent:
        logger.info("scheduled_delivery_pipeline finished sent=%s", sent)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Listen immediately so Railway /health passes; boot DB in the background."""
    if settings.app_env == "production" and (
        settings.secret_key.startswith("biqs-dev")
        or settings.secret_key.startswith("replace-with")
        or len(settings.secret_key) < 24
    ):
        logger.warning("Weak SECRET_KEY in production — rotate before go-live")

    async def _boot() -> None:
        try:
            await init_db()
            _startup_ok["db"] = True
            logger.info("Background DB init OK (backend=%s)", DATABASE_BACKEND)
        except Exception as exc:
            _startup_ok["db"] = False
            _startup_ok["error"] = str(exc)[:300]
            logger.exception(
                "Database init failed — fix DATABASE_URL to Supabase pooler :6543 "
                "(not db.*.supabase.co:5432). Error: %s",
                _startup_ok["error"],
            )
            return

        if supabase_configured():
            try:
                status = await ping_supabase()
                logger.info("Supabase ping: %s", status)
                await ensure_reports_bucket()
            except Exception:
                logger.exception("Supabase startup check failed")
        else:
            logger.warning("Supabase API keys not configured — set SUPABASE_URL + SUPABASE_SECRET_KEY")

        if DATABASE_BACKEND == "sqlite":
            logger.warning("Schedulers disabled on SQLite — use Postgres in production")
            return

        scheduler.add_job(
            scheduled_ai_pipeline,
            "interval",
            hours=settings.scrape_interval_hours,
            id="agency_ai_pipeline",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            scheduled_delivery_pipeline,
            "interval",
            minutes=1,
            id="agency_delivery_pipeline",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        logger.info(
            "MarketBiqs API ready (db=%s, scrape_every=%sh)",
            DATABASE_BACKEND,
            settings.scrape_interval_hours,
        )

    # Critical for Railway: do NOT await DB before yield — otherwise /health is unreachable
    asyncio.create_task(_boot())
    logger.info("MarketBiqs API listening (db_backend=%s, env=%s)", DATABASE_BACKEND, settings.app_env)
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_timing(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)[:200]})
    response.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - started) * 1000))
    return response


# CORS outermost so error responses (401/403/500) still get Access-Control-* headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_origin_regex=r"https://.*\.up\.railway\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Process-Time-Ms"],
)


app.include_router(auth.router, prefix="/api")
app.include_router(agency.router, prefix="/api")
app.include_router(clients.router, prefix="/api")
app.include_router(competitive.router, prefix="/api")
app.include_router(intelligence.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(billing.router, prefix="/api")
app.include_router(integrations.router, prefix="/api")
app.include_router(delivery.router, prefix="/api")
app.include_router(whitelabel.router, prefix="/api")
app.include_router(supabase_api.router, prefix="/api")


@app.get("/health")
async def health():
    """Fast liveness for Railway. Deep checks live under /health/ready."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "database": DATABASE_BACKEND,
        "db_init": _startup_ok["db"],
        "env": settings.app_env,
    }


@app.get("/health/ready")
async def health_ready():
    """Readiness: DB + optional Supabase. Not used by Railway healthcheck."""
    try:
        db_ok = await asyncio.wait_for(ping_db(), timeout=3)
    except Exception:
        db_ok = False
    try:
        supabase = await asyncio.wait_for(ping_supabase(), timeout=3)
    except Exception as exc:
        supabase = {"configured": supabase_configured(), "ok": False, "detail": str(exc)[:200]}
    healthy = db_ok and (supabase.get("ok") if supabase.get("configured") else True)
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "service": settings.app_name,
            "database": DATABASE_BACKEND,
            "database_reachable": db_ok,
            "db_init": _startup_ok["db"],
            "db_init_error": _startup_ok["error"],
            "supabase": supabase,
            "env": settings.app_env,
            "schedulers": ["agency_ai_pipeline", "agency_delivery_pipeline"] if scheduler.running else [],
        },
    )
