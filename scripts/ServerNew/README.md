# alien-server (FastAPI)

Replacement for the legacy PHP alien-server. Provides the HTTP endpoints
(`/createuser`, `/activateuser`, `/login`, ...) that the C++ client in
`source/Network/NetworkService.cpp` talks to.

## Runtime configuration

The application is configured exclusively via environment variables.

| Variable       | Required | Description                                                     |
| -------------- | -------- | --------------------------------------------------------------- |
| `DATABASE_URL` | yes      | SQLAlchemy URL, e.g. `postgresql+psycopg://alien:<pw>@127.0.0.1:5432/alien_db` |

If `DATABASE_URL` is not set, the application refuses to start with a clear
error. This is intentional: shipping a placeholder password would silently
cause `FATAL: password authentication failed for user "alien"` against the
real Postgres instance.

## Local / VPS deployment

The FastAPI app listens on port **8000** by default. Production deployments
put a reverse proxy (nginx / Apache) in front of it on port 80 (or 443 with
TLS) so that the C++ client can reach it at `http://<host>/createuser` etc.

```bash
# 1. Set the database credentials for the alien-server unit (root-only file):
sudo install -m 600 /dev/stdin /etc/alien-server.env <<'EOF'
DATABASE_URL=postgresql+psycopg://alien:<real-password>@127.0.0.1:5432/alien_db
EOF

# 2. Start uvicorn (typically managed by systemd; EnvironmentFile=/etc/alien-server.env):
cd /opt/alien-server
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

If you see `FATAL: password authentication failed for user "alien"` on
startup, the password in `DATABASE_URL` does not match the one configured
in PostgreSQL for the `alien` role. Verify with:

```bash
psql "postgresql://alien:<real-password>@127.0.0.1:5432/alien_db" -c 'select 1;'
```

If that fails, reset the role password:

```bash
sudo -u postgres psql -c "ALTER ROLE alien WITH PASSWORD '<real-password>';"
```

and update `/etc/alien-server.env` accordingly.

## Reverse-proxy example (nginx)

```
server {
    listen 80;
    server_name 85.214.181.38;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
    }
}
```

The C++ client (`source/Base/Resources.h`, `AlienServerURL`) defaults to
`http://85.214.181.38`; `httplib::Client` selects HTTP:80 from the scheme.

## Tests

```bash
pip install -r tests/requirements.txt
pytest scripts/ServerNew/tests
```

The test suite injects its own SQLite-backed `DATABASE_URL` via the
`app_client` fixture, so `DATABASE_URL` does not need to be set in the
shell when running the tests.
