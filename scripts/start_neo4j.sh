#!/usr/bin/env bash
# Start Neo4j in Docker so "Open in Neo4j Browser" works from the Graph view.
# Requires Docker. Browser UI: http://localhost:7474  Bolt: bolt://localhost:7687
# Neo4j 5+ does not allow the default password "neo4j"; we use "neo4j123" unless NEO4J_PASSWORD is set.
# Set the same in apps/api/.env so the API can connect.

set -e
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop (or the Docker daemon), then run this script again."
  echo "Alternatively, install Neo4j without Docker: https://neo4j.com/download/ (Neo4j Desktop) or: brew install neo4j"
  exit 1
fi
IMAGE="${NEO4J_IMAGE:-neo4j:5}"
CONTAINER="${NEO4J_CONTAINER:-anchor-neo4j}"
PASSWORD="${NEO4J_PASSWORD:-neo4j123}"

EXISTING=$(docker ps -a --filter "name=^${CONTAINER}$" --format '{{.Status}}' 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  if echo "$EXISTING" | grep -q "Exited"; then
    echo "Removing exited container ${CONTAINER}..."
    docker rm -f "${CONTAINER}" 2>/dev/null || true
  else
    echo "Container ${CONTAINER} already running."
    echo "  Browser: http://localhost:7474  User: neo4j  Password: ${PASSWORD}"
    exit 0
  fi
fi
echo "Creating and starting Neo4j container ${CONTAINER}..."
docker run -d \
    --name "${CONTAINER}" \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/"${PASSWORD}" \
    "${IMAGE}"

echo ""
echo "Neo4j is starting. In a few seconds:"
echo "  - Browser UI:  http://localhost:7474/browser"
echo "  - Bolt:        bolt://localhost:7687"
echo "  - User:        neo4j"
echo "  - Password:    ${PASSWORD}"
echo ""
echo "To match your existing .env password, run with that value first (then recreate container if needed):"
echo "  NEO4J_PASSWORD=yourpassword ./scripts/start_neo4j.sh"
echo "Set the same in apps/api/.env: NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=..."
echo ""
echo "Neo4j 5 does not allow password 'neo4j'. To use that, run: NEO4J_IMAGE=neo4j:4 NEO4J_PASSWORD=neo4j ./scripts/start_neo4j.sh"
echo "Then restart the API and use Graph view → Sync to Neo4j → Open in Neo4j Browser."
