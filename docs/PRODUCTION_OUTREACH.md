# Production outreach setup

## New endpoints added

| Method | Path | Description |
|--------|------|-------------|
| POST | /actions/outreach | Trigger caregiver outreach (body: risk_signal_id, channel_preference?, dry_run?). Caregiver/admin only. |
| GET | /actions/outreach | List recent outbound actions for household. Elder sees elder_safe only. |
| GET | /actions/outreach/{id} | Get one outbound action. Elder sees elder_safe_message + status only. |
| GET | /actions/outreach/summary | Counts (sent/suppressed/failed/queued/delivered) + recent list. Caregiver/admin only. |
| POST | /agents/outreach/run | Run Caregiver Outreach Agent (body: risk_signal_id, dry_run?). Caregiver/admin only. |

## New env vars required

| Variable | Description | Required for |
|----------|-------------|--------------|
| ANCHOR_NOTIFY_PROVIDER | `mock` \| `twilio` \| `sendgrid` \| `smtp` | All (default: mock) |
| TWILIO_ACCOUNT_SID | Twilio account SID | Twilio SMS |
| TWILIO_AUTH_TOKEN | Twilio auth token | Twilio SMS |
| TWILIO_FROM | From phone number (e.g. +1…) | Twilio SMS |
| SENDGRID_API_KEY | SendGrid API key | SendGrid email |
| SENDGRID_FROM | From email address | SendGrid email |
| SMTP_HOST | SMTP server host | SMTP email |
| SMTP_USER | SMTP username (optional) | SMTP email |
| SMTP_PASSWORD | SMTP password (optional) | SMTP email |
| SMTP_FROM | From email address | SMTP email |
| ANCHOR_WORKER_OUTREACH_AUTO_TRIGGER | `true` \| `false` (default: true) | Worker auto-trigger after persist |

## Migrations to run

Run in order (after 001–012):

- **db/migrations/011_role_consent_helpers.sql** — `user_can_contact()` helper
- **db/migrations/012_outbound_actions_caregiver_contacts.sql** — outbound_actions, caregiver_contacts tables + RLS
- **db/migrations/013_outbound_contact_safe_display.sql** — recipient_contact_last4 column

## Frontend components changed

- **apps/web/src/app/(dashboard)/alerts/[id]/alert-detail-content.tsx** — Notify caregiver button, preview/confirm, delivery status + timestamps + error
- **apps/web/src/app/(dashboard)/agents/page.tsx** — Outreach Agent card: counts (sent/suppressed/failed) + recent list; useOutreachSummary
- **apps/web/src/app/(dashboard)/elder/page.tsx** — elder_safe_message + “Caregiver notified” indicator (consent-gated)
- **apps/web/src/lib/api/schemas.ts** — OutreachActionSchema, OutreachSummarySchema
- **apps/web/src/lib/api/client.ts** — getOutreachSummary()
- **apps/web/src/hooks/use-api.ts** — useOutreachSummary(enabled?)

---

## Human-required inputs

- [ ] **Notify provider**: Set `ANCHOR_NOTIFY_PROVIDER` to `mock` (default), `twilio`, `sendgrid`, or `smtp`.
- [ ] **Twilio (SMS)**: If using Twilio, supply:
  - [ ] `TWILIO_ACCOUNT_SID` (from Twilio console)
  - [ ] `TWILIO_AUTH_TOKEN` (from Twilio console)
  - [ ] `TWILIO_FROM` (e.g. +1234567890)
- [ ] **SendGrid (email)**: If using SendGrid, supply:
  - [ ] `SENDGRID_API_KEY` (from SendGrid API keys)
  - [ ] `SENDGRID_FROM` (verified sender email)
- [ ] **SMTP (email fallback)**: If using SMTP, supply:
  - [ ] `SMTP_HOST` (required)
  - [ ] `SMTP_FROM` (required)
  - [ ] `SMTP_USER` / `SMTP_PASSWORD` (if auth required)
- [ ] **Worker**: Set `ANCHOR_WORKER_OUTREACH_AUTO_TRIGGER=false` to disable auto outreach after pipeline persist (default: true).
- [ ] **Migrations**: Run 011, 012, 013 on your Supabase (or Postgres) database.
- [ ] **Caregiver contacts**: Populate `caregiver_contacts` (household_id, name, channels, priority) for each household so outreach has a recipient; otherwise demo placeholder is used.
- [ ] **Consent**: Ensure `sessions.consent_state.consent_allow_outbound_contact` is set (e.g. via UI or API) when the elder opts in to outbound contact; default in code is false (opt-in).
