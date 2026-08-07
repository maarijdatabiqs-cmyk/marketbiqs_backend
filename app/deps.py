from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Agency, AgencyMember, ClientBrand, User, WhiteLabelApiKey
from app.security import decode_access_token, hash_api_key

bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    user: User
    agency: Agency
    membership: AgencyMember


def _email_from_claims(payload: dict) -> str:
    email = (payload.get("email") or "").strip().lower()
    if email:
        return email
    meta = payload.get("user_metadata") or {}
    email = (meta.get("email") or "").strip().lower()
    if email:
        return email
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing email")


def _name_from_claims(payload: dict, email: str) -> str:
    meta = payload.get("user_metadata") or {}
    for key in ("full_name", "name", "preferred_username"):
        value = (meta.get(key) or "").strip()
        if value:
            return value[:255]
    return email.split("@")[0][:255] or "User"


async def _ensure_user_row(
    db: AsyncSession,
    *,
    user_id: str,
    email: str,
    full_name: str,
) -> User:
    """Create app user for Auth UUID; tolerate parallel /me races."""
    user = await db.get(User, user_id)
    if user:
        return user

    # Fresh Auth user — claim email if a legacy row still holds it.
    legacy = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if legacy and legacy.id != user_id:
        memberships = (
            await db.execute(select(AgencyMember).where(AgencyMember.user_id == legacy.id))
        ).scalars().all()
        for membership in memberships:
            membership.user_id = user_id
        await db.delete(legacy)
        await db.flush()
        user = await db.get(User, user_id)
        if user:
            return user

    user = User(
        id=user_id,
        email=email,
        full_name=full_name,
        hashed_password=None,
    )
    try:
        async with db.begin_nested():
            db.add(user)
            await db.flush()
    except IntegrityError:
        # Parallel /api/auth/me (or bootstrap) already inserted this Auth user.
        existing = await db.get(User, user_id)
        if existing:
            return existing
        by_email = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if by_email:
            return by_email
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not create user account — please try again",
        ) from None
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")

    email = _email_from_claims(payload)
    full_name = _name_from_claims(payload, email)

    user = await _ensure_user_row(db, user_id=user_id, email=email, full_name=full_name)

    if email and user.email != email:
        conflict = (
            await db.execute(select(User).where(User.email == email, User.id != user_id))
        ).scalar_one_or_none()
        if conflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already linked to another account",
            )
        user.email = email
    if full_name and (not user.full_name or user.full_name == user.email.split("@")[0]):
        user.full_name = full_name
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    await db.flush()
    return user


async def get_auth_context(
    user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
    x_agency_id: str | None = Header(default=None, alias="X-Agency-Id"),
) -> AuthContext:
    stmt = (
        select(AgencyMember)
        .options(selectinload(AgencyMember.agency), selectinload(AgencyMember.user))
        .where(AgencyMember.user_id == user.id, AgencyMember.is_active.is_(True))
    )
    result = await db.execute(stmt)
    memberships = list(result.scalars().all())
    if not memberships:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No agency membership")

    # Prefer JWT claim (legacy), then header — ignore stale X-Agency-Id from a previous session.
    jwt_agency_id: str | None = None
    if credentials:
        try:
            jwt_agency_id = decode_access_token(credentials.credentials).get("agency_id")
        except ValueError:
            jwt_agency_id = None

    preferred = jwt_agency_id or x_agency_id
    membership = memberships[0]
    if preferred:
        match = next((m for m in memberships if m.agency_id == preferred), None)
        if match:
            membership = match
    return AuthContext(user=user, agency=membership.agency, membership=membership)


async def require_roles(*roles: str):
    async def checker(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if ctx.membership.role.value not in roles and ctx.membership.role.value != "owner":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return ctx

    return checker


async def get_tenant_client(
    client_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> ClientBrand:
    client = await db.get(ClientBrand, client_id)
    if not client or client.agency_id != ctx.agency.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    return client


async def get_white_label_agency(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Agency:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
    hashed = hash_api_key(x_api_key)
    stmt = select(WhiteLabelApiKey).where(
        WhiteLabelApiKey.hashed_key == hashed,
        WhiteLabelApiKey.is_active.is_(True),
    )
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if record.requests_used >= record.monthly_quota:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Quota exceeded")
    agency = await db.get(Agency, record.agency_id)
    if not agency:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Agency missing")
    record.requests_used += 1
    await db.flush()
    return agency
