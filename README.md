# Aura API

**Adaptive User Risk Analyzer** — a FastAPI service intended to analyze emails for phishing detection and adaptive risk scoring.

> **Current state:** the repository ships the auth/user-management scaffold only. The phishing-analysis (`analysis`) module described in `CLAUDE.md` has not yet been built. What is in the repo is documented below; references in `CLAUDE.md` to `src/modules/`, the `analysis` module, the `Classification` enum, and `analysis.*` config are forward-looking.

---

## What this currently is

A FastAPI service with:

- JWT auth (access + refresh) backed by a `tokens` table for revocation
- Bcrypt password hashing
- Role-based access control (`ADMIN`, `IT_ANALYST`)
- Admin-only user management endpoints (CRUD, activate/deactivate, reset password)
- Self-service profile + password-change endpoints
- A startup admin seeder, a background expired-token cleanup loop, and `/health` + `/ready` ops endpoints
- A bespoke YAML+env config system with IDE stub generation
- A consistent `ApiResponse` / `PaginatedResponse` envelope and global exception → envelope handlers
- IP-based rate limiting on auth endpoints (slowapi)
- Alembic migrations for the auth schema

---

## Project structure

```
aura_api/
├── main.py                     # uvicorn entrypoint (factory mode)
├── alembic.ini                 # alembic config (DB URL is overridden by env.py)
├── alembic/
│   ├── env.py                  # imports DATABASE_URL from src.configs
│   └── versions/               # 0001_init_schema, 0002_token_hash
├── requirements.txt
├── .env.example
└── src/
    ├── configs/                # YAML+env loader + generated .pyi stub
    │   ├── application.yaml    # source of truth for all config
    │   ├── loader.py
    │   └── generate.py
    ├── shared/                 # generic kernel — no app-specific code
    │   ├── database/           # engine, session, BaseModel, BaseRepository, pagination
    │   ├── responses/          # ApiResponse, PaginatedResponse, ErrorDetail
    │   └── exceptions/         # AppException hierarchy + global handlers
    ├── core/                   # app composition
    │   ├── factory.py          # create_app()
    │   ├── lifespan.py         # startup/shutdown — seeder + cleanup task
    │   ├── middleware.py       # CORS + raw-ASGI request logging w/ X-Request-ID
    │   └── rate_limit.py       # slowapi limiter
    └── app/                    # auth + user management (layered, not per-module)
        ├── controllers/        # auth_controller, user_controller, system_controller
        ├── services/           # AuthService, UserManagementService, password_hasher,
        │                       # token_provider, admin_seeder, token_cleanup
        ├── repositories/       # UserRepository, TokenRepository
        ├── models/             # User, Token + Role/TokenType enums
        ├── dtos/               # request + response Pydantic models
        └── dependencies.py     # bearer_scheme, current-user + role guards, service factories
```

The `src/modules/` directory referenced in `CLAUDE.md` does not currently exist.

---

## Setup

Requires Python 3.11+ and a PostgreSQL instance.

```bash
# 1. Create + activate a virtualenv
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # POSIX

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy + edit env
cp .env.example .env           # POSIX
copy .env.example .env         # Windows
# At minimum set DATABASE_URL, JWT_SECRET_KEY, ADMIN_PASSWORD

# 4. Apply migrations
alembic upgrade head

# 5. Run the dev server
python main.py
```

The dev server binds to `127.0.0.1:8000`. Reload is gated on `DEBUG=true`.

### Generating a JWT secret

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Alembic notes

`alembic/env.py` resolves `DATABASE_URL` via `src.configs`, so the value in `alembic.ini` is ignored at runtime. To run migrations, you only need `DATABASE_URL` set — `JWT_SECRET_KEY` and `ADMIN_PASSWORD` may be left blank (their defaults are empty; they are validated only when the auth code or seeder actually runs).

---

## Environment variables

All variables have sensible defaults except where marked **required**. See `.env.example` for the canonical list.

### Application
| Var | Default | Notes |
|---|---|---|
| `APP_NAME` | `AuraAPI` | |
| `APP_VERSION` | `0.1.0` | |
| `DEBUG` | `false` | Enables FastAPI debug mode and uvicorn reload |
| `ENVIRONMENT` | `development` | One of `development`, `staging`, `production`. Stub regen runs only in `development`. |

### Database
| Var | Default | Notes |
|---|---|---|
| `DATABASE_URL` | — | **Required.** Must use `postgresql+asyncpg://…` |
| `DB_POOL_SIZE` | `5` | |
| `DB_MAX_OVERFLOW` | `10` | |
| `DB_POOL_TIMEOUT` | `30` | |
| `DB_ECHO` | `false` | |

### Security — JWT
| Var | Default | Notes |
|---|---|---|
| `JWT_SECRET_KEY` | _empty_ | Required at runtime when any auth code runs (not for migrations). |
| `JWT_ALGORITHM` | `HS256` | |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | |

### Security — admin seeder
| Var | Default | Notes |
|---|---|---|
| `ADMIN_EMAIL` | `admin@aura.local` | |
| `ADMIN_USERNAME` | `admin` | |
| `ADMIN_FIRST_NAME` | `System` | |
| `ADMIN_LAST_NAME` | `Admin` | |
| `ADMIN_PASSWORD` | _empty_ | If empty, admin seeding is skipped (server still boots). |

### Security — password hashing & token cleanup
| Var | Default | Notes |
|---|---|---|
| `BCRYPT_ROUNDS` | `12` | bcrypt cost factor |
| `TOKEN_CLEANUP_INTERVAL_SECONDS` | `3600` | Background loop interval |

### Server — CORS
| Var | Default | Notes |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated. **Cannot include `*` if `CORS_ALLOW_CREDENTIALS=true`** — startup will abort. |
| `CORS_ALLOW_CREDENTIALS` | `true` | |
| `CORS_ALLOW_METHODS` | `*` | |
| `CORS_ALLOW_HEADERS` | `*` | |

### Server — rate limiting (slowapi)
| Var | Default | Notes |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | |
| `RATE_LIMIT_LOGIN` | `5/minute` | Format `{count}/{period}` |
| `RATE_LIMIT_REGISTER` | `3/minute` | |
| `RATE_LIMIT_REFRESH` | `10/minute` | |

### Logging
| Var | Default | Notes |
|---|---|---|
| `LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FILE_PATH` | `logs/app.log` | Directory is created if missing |

---

## API endpoints

All responses use the `ApiResponse` / `PaginatedResponse` envelope. Successful responses carry `value`; failures carry `error` (a structured `ErrorDetail`). JSON keys are camelCase.

Auth header for protected routes:

```
Authorization: Bearer <access_token>
```

### System (no auth)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Process liveness — returns app name + version |
| `GET` | `/ready` | DB connectivity check (`SELECT 1`) |
| `POST` | `/api/v1/system/reload-config` | **Admin only.** Re-reads `application.yaml` |

### Auth — `/api/v1/auth`

| Method | Path | Auth | Rate limit | Description |
|---|---|---|---|---|
| `POST` | `/register` | public | `RATE_LIMIT_REGISTER` | Create user (defaults to `IT_ANALYST`); returns user + token pair |
| `POST` | `/login` | public | `RATE_LIMIT_LOGIN` | Email + password; revokes all prior tokens before issuing a new pair |
| `POST` | `/refresh` | public | `RATE_LIMIT_REFRESH` | Refresh-token rotation; revokes all prior tokens |
| `POST` | `/logout` | bearer | — | Revokes all of the caller's tokens |
| `GET` | `/me` | bearer | — | Current user profile |
| `PATCH` | `/me` | bearer | — | Update first/last name and/or username |
| `PATCH` | `/me/password` | bearer | — | Change password (current + new); revokes all sessions |

### Users — `/api/v1/users` (admin only)

| Method | Path | Description |
|---|---|---|
| `GET` | `` | Paginated list; filter by `role`, `isActive` |
| `GET` | `/{user_id}` | Fetch by id |
| `POST` | `` | Create user (admin can pick role) |
| `PATCH` | `/{user_id}` | Update profile / role / active flag (last-admin guard applies) |
| `DELETE` | `/{user_id}` | Hard delete (last-admin guard applies, cascades tokens) |
| `POST` | `/{user_id}/activate` | Set `is_active=true` |
| `POST` | `/{user_id}/deactivate` | Set `is_active=false` (last-admin guard, revokes tokens) |
| `POST` | `/{user_id}/reset-password` | Force-set new password; revokes tokens |

---

## Architectural conventions

A contributor needs to know:

### Config
- `src/configs/application.yaml` is the source of truth. Format: `"${ENV:default} | type"`, `"${ENV} | type | required"`, or `"value | type"`. Types: `str`, `int`, `float`, `bool`, `list`.
- On import, `src/configs/__init__.py` loads the YAML, resolves env vars, validates required fields (collecting every error before raising), and exposes each top-level section as a module attribute (`from src.configs import database; database.url`).
- The `.pyi` IDE stub is regenerated only when `ENVIRONMENT=development`.
- When adding a config section, update `application.yaml` and `.env.example` together.

### Database & repositories
- `BaseModel` (in `src.shared.database`) supplies `id` (UUID, server-side `gen_random_uuid()`), `created_at`, `updated_at` on every table.
- Enums inherit `(str, enum.Enum)` and are persisted with `SQLAlchemy Enum(..., name="{field}_enum")`.
- `get_db` opens `session.begin()` — commits on clean return, rolls back on exception. `get_db_no_transaction` skips the outer transaction (caller manages).
- Repository write methods call `flush` (never `commit`) — the dependency owns the transaction.
- `BaseRepository[T]` provides `get_by_id`, `get_one`, `exists`, `get_all`, `paginate`, `count`, `create`, `create_many(refresh=False)`, `update`, `delete`. Use it for equality filters; write custom `select()` in subclasses for `ilike`, joins, complex ordering.

### Responses & errors
- Always return `ApiResponse.ok(value=..., message=...)` or `PaginatedResponse.ok(value=..., page=..., total=..., page_size=...)`. `value` and `error` are mutually exclusive (validator-enforced).
- JSON output uses `model_dump(exclude_none=True, by_alias=True)`. Response DTOs declare camelCase aliases (`Field(alias="createdAt")`) and `Config: populate_by_name = True; from_attributes = True`, with a `from_{entity}(entity)` static factory.
- Raise an `AppException` subclass (`NotFoundException`, `ConflictException`, `BadRequestException`, `AuthenticationException`, `AuthorizationException`, `ValidationException`, `InternalServerException`, `ServiceUnavailableException`) with an `ErrorDetail`. Use `ErrorDetail.builder(...).add_field_error(...)` for per-field errors.
- Global handlers in `src/shared/exceptions/error_handlers.py` translate to the envelope. There is also a dedicated `IntegrityError` → 409 handler.

### Auth invariants
- JWT pair is created on register/login/refresh; each token is persisted as a SHA-256 hex hash in the `tokens` table. `verify_token` decodes the JWT *and* checks the DB row is not revoked/expired.
- `login` and `refresh_token` revoke **all** prior user tokens before issuing the new pair (refresh-token rotation).
- `logout` revokes all of the caller's tokens (uses `verify_token`, not bare JWT decode).
- Roles: `ADMIN`, `IT_ANALYST`. Use `require_admin` / `require_authenticated` / `require_role(...)` from `src.app.dependencies`.
- A last-admin guard prevents demoting, deactivating, or deleting the only active admin.

### Middleware
- `RequestLoggingMiddleware` is raw ASGI on purpose — `BaseHTTPMiddleware` buffers response bodies and breaks SSE. Keep any new middleware raw ASGI if it may touch streaming.
- Every request gets an `X-Request-ID` (incoming or generated) attached to the response and to log lines (`[rid=…]`).

### Lifespan & background tasks
- `src/core/lifespan.py` sets up logging, runs `seed_admin()` (creates an ADMIN from `security.admin.*` env if none exists), and spawns `start_token_cleanup()` (purges expired JWTs on `TOKEN_CLEANUP_INTERVAL_SECONDS`; runs once immediately on startup so a backlog clears without waiting a full interval).
- New background tasks: `asyncio.create_task(...)` before `yield`, cancel + `await` with `CancelledError` suppression after `yield`. The loop itself must catch `CancelledError` and re-raise.

### Cross-layer rules
- Bounded-context modules under `src/modules/` (when they exist) may import from `src/app/` only these primitives: `get_current_user`, `require_role`, `require_admin`, `require_authenticated`, `Role`, `User`. Never import an app service, repository, or DTO from a module.

---

## Commands

```bash
# Run the dev server (factory + reload gated on DEBUG)
python main.py

# Alembic
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
alembic downgrade -1

# Regenerate the IDE config stub (boot does this automatically in development)
python -m src.configs.generate
```

No test suite, linter, or formatter is configured.
