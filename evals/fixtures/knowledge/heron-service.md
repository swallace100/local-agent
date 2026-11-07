# Heron ingest service

Heron receives sensor batches from field gateways and writes them to cold storage.

- **Port:** the service listens on port 7421.
- **Owner:** Marisol Quint runs on-call and approves changes.
- **Deploys:** Tuesdays and Thursdays, never during a batch window.

Batches over 40 MB are rejected with a 413 response. Gateways retry after 90 seconds.
