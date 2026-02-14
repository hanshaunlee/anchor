# Neo4j setup (optional)

Neo4j is **optional**. The Anchor app works without it. When enabled, the **Graph view** in the dashboard can sync the household evidence graph to Neo4j so you can run Cypher queries and use Neo4j Browser for exploration.

## Quick start (Docker)

**Start Docker first** (Docker Desktop or your system’s Docker daemon). If you see “Cannot connect to the Docker daemon”, Docker isn’t running. On macOS you can start it with:
`open -a Docker`
Wait for the whale icon in the menu bar to stop animating, then run the script below.

From the repo root:

```bash
./scripts/start_neo4j.sh
```

This starts Neo4j in a container. Default password is `neo4j123` (Neo4j 5 does not allow `neo4j`). Then:

1. Add to `apps/api/.env`: `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=neo4j123`
2. Restart the API
3. Open **Graph view** → **Sync to Neo4j** → **Open in Neo4j Browser**

The link pre-fills the query (click **Run** once) and, when the API has `NEO4J_PASSWORD` set, includes the connection URL with credentials so Neo4j Browser can pre-fill or connect without the user typing anything.

If you see “site can’t be reached”, wait a few seconds for Neo4j to finish starting, or run `docker logs anchor-neo4j` to confirm it’s up.

**Match an existing password:** Run `NEO4J_PASSWORD=yourpassword ./scripts/start_neo4j.sh` (then remove the old container if one exists: `docker rm -f anchor-neo4j`). Set the same in `apps/api/.env`. **Neo4j 5 does not allow the password `neo4j`**; to use that, run with Neo4j 4: `NEO4J_IMAGE=neo4j:4 NEO4J_PASSWORD=neo4j ./scripts/start_neo4j.sh`.

**No Docker?** Install [Neo4j Desktop](https://neo4j.com/download/) (GUI, no Docker needed) or, on macOS, `brew install neo4j` then `neo4j console`. Use the same `.env` settings; Browser is at http://localhost:7474/browser when Neo4j is running.

## When Neo4j is used

- **Graph view** in the web app: "Sync to Neo4j" and "Open in Neo4j Browser" appear only when Neo4j is configured.
- Sync runs on demand (user clicks "Sync to Neo4j"); it does not run automatically.
- The ML pipeline and risk engine do **not** use Neo4j; they use Supabase and in-memory graph building.

## How to ensure Neo4j works

### 1. Install and run Neo4j

- **Desktop**: [Neo4j Desktop](https://neo4j.com/download/) — create a local DB and start it.
- **Docker**: `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j`
- **AuraDB**: Create a free instance at [neo4j.com/cloud](https://neo4j.com/cloud/) and use its Bolt URI.

### 2. Configure the API

Set these in the API `.env` (e.g. `apps/api/.env` or repo root `.env`):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword
```

- **Bolt** is required (e.g. `bolt://localhost:7687`). HTTP (7474) is for the Browser UI only.
- If `NEO4J_URI` is empty or unset, Neo4j is disabled and the UI will not show sync/browser buttons.

### 3. Install the Python driver (API)

The API uses the official `neo4j` driver. Install it in the API environment:

```bash
pip install neo4j
```

If the package is missing, the API will log that Neo4j sync is disabled and will not crash.

### 4. Verify from the UI

1. Start the API with the env vars above.
2. Open the dashboard → **Graph view**.
3. You should see **Sync to Neo4j** and **Open in Neo4j Browser** when Neo4j is enabled.
4. Click **Sync to Neo4j** — the current household graph (entities + relationships) is written to Neo4j.
5. Click **Open in Neo4j Browser** (for local Neo4j this opens `http://localhost:7474`). Run a query, e.g.:

   ```cypher
   MATCH (e:Entity) RETURN e LIMIT 25
   ```

### 5. Verify from the API

- **Status**: `GET /graph/neo4j-status` returns `{"enabled": true, "browser_url": "http://localhost:7474"}` when Neo4j is configured (and local). No auth required.
- **Sync**: `POST /graph/sync-neo4j` (with auth) runs the sync; response includes `ok`, `message`, `entities`, `relationships`.

## Troubleshooting

| Symptom | Check |
|--------|--------|
| "Neo4j not configured" | `NEO4J_URI` set in API `.env` and API restarted. |
| Sync fails / connection error | Neo4j running; correct Bolt port (7687); firewall; `NEO4J_USER`/`NEO4J_PASSWORD` correct. |
| No "Open in Neo4j Browser" | API only adds `browser_url` for localhost/127.0.0.1. For remote Neo4j, open your instance URL manually. |
| Driver not found | `pip install neo4j` in the API environment. |
